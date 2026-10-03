"""
orchestration/telemetry.py

Observability for the router, built on the extension points LangChain and
LangGraph already expose -- not hand-wrapped call sites. This replaces an
earlier version of this file that manually decorated every node and
wrapped every `model.invoke()` call; that approach worked but duplicated
machinery LangChain already ships. Verified empirically (see commit
history / dev notes) that LangGraph propagates callbacks AND node
identity to every nested node and every `model.invoke()` call
automatically via contextvars, so a single handler attached once at
`app.invoke(config={"callbacks": [...]})` sees everything -- no `config=`
threading through classify_subquestion / run_sql / run_compute, no
decorator on every node.

Three pillars, kept separate on purpose -- this split is the standard
shape for production LLM systems, not something invented for this repo:

1. TRACES + METRICS (this file). Cost, tokens, latency, errors, and node
   graph shape, captured via `RouterCallbackHandler` and emitted as
   OpenTelemetry -- the vendor-neutral instrumentation standard. Point
   OTEL_EXPORTER_OTLP_ENDPOINT at an OTel Collector in front of
   Honeycomb / Datadog / Grafana Tempo / CloudWatch and nothing in this
   file changes. Defaults to a console exporter so it runs with zero
   infra in dev.

2. PROMPT/COMPLETION CONTENT -- deliberately NOT captured here. Full
   prompt/completion text, side-by-side run diffing, and dataset-based
   eval is what LangSmith/Langfuse are actually built for; a metrics
   stack is the wrong tool for reading LLM traffic. Both now accept
   OTel-native ingestion (Langfuse ships an OTLP endpoint; LangSmith has
   its own SDK that also hooks the same callback system), so the spans
   this file emits can be dual-exported there instead of re-instrumenting.
   See `wire_langfuse()` below for the one-line version.

3. BUSINESS / AUDIT LOG (`log_decision`). Blocked queries, retry reasons,
   cache hits, HITL flags. Not performance data -- these are the
   decisions a reviewer or auditor asks about later ("why was this
   query blocked", "how often do we fall back to the web"). Kept as a
   thin structured JSONL now; moves to a DB table once Postgres exists,
   not into the metrics backend -- audit trails and dashboards have
   different retention/query needs.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

AUDIT_LOG_PATH = Path("data/audit_log.jsonl")
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

SERVICE_NAME = "filingsiq-router"

# gpt-4o-mini, confirmed current as of Aug 2026: https://openai.com/api/pricing
# Re-check periodically -- this is a config value, not a constant to trust forever.
PRICING = {
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}
DEFAULT_MODEL_NAME = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# OTel wiring. OTLP endpoint set via the standard OTEL_EXPORTER_OTLP_ENDPOINT
# env var -- if unset, falls back to console export so this works with zero
# infra locally. Swapping to a real collector in prod is an env var, not a
# code change.
# ---------------------------------------------------------------------------

_resource = Resource.create({"service.name": SERVICE_NAME})


def _quiet() -> bool:
    # Confirmed a real, recurring friction point: eval scripts kept
    # needing a manual `2>/dev/null` to stay readable, and that trick
    # only works because ConsoleSpanExporter happens to write where
    # redirection catches it -- fragile to rely on forever. This is the
    # explicit, permanent way to turn console export off for a clean
    # local run without needing a real OTLP collector either.
    return os.environ.get("OTEL_QUIET", "").lower() in ("1", "true", "yes")


def _phoenix_span_processor() -> Optional[BatchSpanProcessor]:
    """
    Dual-exports the SAME spans to Arize Phoenix, as an ADDITIONAL
    processor on the one TracerProvider this file already builds --
    deliberately not a second, competing provider. Phoenix's own
    `register()` convenience helper sets up its own TracerProvider
    internally; calling that here would collide with the provider this
    file already configures below, since OTel's global provider is
    effectively set-once per process. Attaching one more span processor
    to the existing provider avoids that collision entirely and matches
    this file's own stated design (env var in, no code change) for every
    other backend.

    Reads the same env var names Phoenix's own docs use, so this stays
    compatible if the official arize-phoenix-otel package is adopted later:
      PHOENIX_COLLECTOR_ENDPOINT -- e.g. http://localhost:6006 (self-hosted)
                                     or https://app.phoenix.arize.com/s/<space> (cloud)
      PHOENIX_API_KEY            -- required for cloud, unused for local
      PHOENIX_OTLP_PROTOCOL      -- "grpc" for local self-hosted (port 4317,
                                     Phoenix's docs call this "more
                                     performant"); defaults to http, which
                                     Phoenix's docs call "simpler" and is
                                     the safer default for a remote/cloud
                                     endpoint over standard HTTPS.
    """
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return None
    api_key = os.environ.get("PHOENIX_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if os.environ.get("PHOENIX_OTLP_PROTOCOL", "http").lower() == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
        exporter = GrpcExporter(endpoint=endpoint, headers=headers or None, insecure=not api_key)
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpExporter
        # Phoenix serves OTLP/HTTP at /v1/traces specifically -- not the
        # generic OTLP-standard path -- so append it if the endpoint given
        # doesn't already include it, letting PHOENIX_COLLECTOR_ENDPOINT be
        # set exactly as Phoenix's own docs/UI display it, verbatim.
        traces_url = endpoint if endpoint.rstrip("/").endswith("/v1/traces") else endpoint.rstrip("/") + "/v1/traces"
        exporter = HttpExporter(endpoint=traces_url, headers=headers or None)

    return BatchSpanProcessor(exporter)


def _build_tracer_provider() -> TracerProvider:
    provider = TracerProvider(resource=_resource)
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    elif not _quiet():
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    # else: OTEL_QUIET set, OTLP not configured -- no exporter at all.
    # Spans/metrics are still recorded internally (nothing about the
    # router's own logic changes), they just have nowhere to print.

    # Additive and independent of the branch above -- Phoenix receives
    # spans regardless of whether a separate OTLP collector or console
    # export is also configured, since this attaches its own processor
    # rather than replacing whatever was chosen above.
    phoenix_processor = _phoenix_span_processor()
    if phoenix_processor is not None:
        provider.add_span_processor(phoenix_processor)

    return provider


def _build_meter_provider() -> MeterProvider:
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        reader = PeriodicExportingMetricReader(OTLPMetricExporter())
        return MeterProvider(resource=_resource, metric_readers=[reader])
    if _quiet():
        return MeterProvider(resource=_resource, metric_readers=[])
    reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60_000)
    return MeterProvider(resource=_resource, metric_readers=[reader])


trace.set_tracer_provider(_build_tracer_provider())
metrics.set_meter_provider(_build_meter_provider())

tracer = trace.get_tracer(SERVICE_NAME)
meter = metrics.get_meter(SERVICE_NAME)

llm_cost_usd = meter.create_counter("llm.cost.usd", unit="usd", description="LLM spend by node/model")
llm_tokens = meter.create_counter("llm.tokens", description="Token count by node/direction")
node_latency_ms = meter.create_histogram("router.node.latency", unit="ms", description="Per-node wall time")
llm_latency_ms = meter.create_histogram("router.llm.latency", unit="ms", description="Per-LLM-call wall time")
retry_count_ctr = meter.create_counter("router.retries", description="Retry events by reason")
blocked_count_ctr = meter.create_counter("router.blocked", description="Blocked queries by reason")
faithfulness_hist = meter.create_histogram("router.faithfulness", description="Faithfulness score distribution")
cache_hit_ctr = meter.create_counter("router.cache", description="Cache hit/miss")


def wire_langfuse() -> Optional[BaseCallbackHandler]:
    """
    Optional: return a Langfuse callback handler if LANGFUSE_PUBLIC_KEY is
    set, so prompt/completion content also lands in Langfuse's UI without
    a second instrumentation pass. Call this once in answer_question() and
    append the result to the callbacks list if not None. Not wired in by
    default -- content capture is opt-in given it stores full prompts.
    """
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    from langfuse.callback import CallbackHandler  # pip install langfuse

    return CallbackHandler()


# ---------------------------------------------------------------------------
# Audit log -- business decisions only, never prompt/completion content.
# ---------------------------------------------------------------------------


def log_decision(event: str, *, trace_id: str, **fields: Any) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "trace_id": trace_id,
        **fields,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# The callback handler. Attach ONE instance per request
# (config={"callbacks": [handler]} on app.invoke) -- LangGraph propagates
# it to every node and every model.invoke() inside them automatically.
# ---------------------------------------------------------------------------


class RouterCallbackHandler(BaseCallbackHandler):
    # Content capture is opt-in, not default -- full prompt/completion
    # text is a different storage/PII tradeoff than pure metrics, same
    # reasoning as wire_langfuse() being opt-in. Auto-enables the moment a
    # Phoenix endpoint is configured, since there's no reason to make
    # someone set two separate env vars for one feature -- an explicit
    # OTEL_CAPTURE_LLM_CONTENT=false still overrides this if metrics-only
    # is wanted even with Phoenix configured.
    CAPTURE_LLM_CONTENT = (
        os.environ.get("OTEL_CAPTURE_LLM_CONTENT", "").lower() == "true"
        or (bool(os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")) and os.environ.get("OTEL_CAPTURE_LLM_CONTENT", "").lower() != "false")
    )
    _MAX_CONTENT_CHARS = 4000  # guard against pathologically long spans (e.g. a huge vector context)

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self._chain_starts: dict[str, tuple[float, str]] = {}  # run_id -> (t0, node_name)
        self._llm_starts: dict[str, tuple[float, str, str]] = {}  # run_id -> (t0, node_name, prompt)
        self.llm_calls: list[dict] = []

    # ---- per-node latency, keyed off LangGraph's own node metadata ----
    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kw):
        node = (metadata or {}).get("langgraph_node")
        if node:
            self._chain_starts[str(run_id)] = (time.monotonic(), node)

    def on_chain_end(self, outputs, *, run_id, **kw):
        entry = self._chain_starts.pop(str(run_id), None)
        if entry is None:
            return
        t0, node = entry
        latency = (time.monotonic() - t0) * 1000
        node_latency_ms.record(latency, {"node": node})
        with tracer.start_as_current_span(f"node.{node}", attributes={"trace_id": self.trace_id}) as span:
            span.set_attribute("latency_ms", latency)

    def on_chain_error(self, error, *, run_id, **kw):
        entry = self._chain_starts.pop(str(run_id), None)
        node = entry[1] if entry else "unknown"
        log_decision("node_error", trace_id=self.trace_id, node=node, error=str(error))

    # ---- per-LLM-call cost/tokens ----
    # NOTE: on_llm_end is NOT guaranteed to receive `metadata` the way
    # on_llm_start is (verified empirically -- it was consistently missing
    # in testing even though on_llm_start's metadata reliably contains
    # langgraph_node). So the node name is captured at start and looked
    # up by run_id at end, rather than trusted from on_llm_end directly.
    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kw):
        node = (metadata or {}).get("langgraph_node", "unknown")
        prompt_text = prompts[0] if prompts else ""
        self._llm_starts[str(run_id)] = (time.monotonic(), node, prompt_text)

    def on_llm_end(self, response, *, run_id, metadata=None, **kw):
        entry = self._llm_starts.pop(str(run_id), None)
        t0, node, prompt_text = entry if entry else (None, "unknown", "")
        latency = (time.monotonic() - t0) * 1000 if t0 else None

        usage = (response.llm_output or {}).get("token_usage")
        if not usage:
            gen_msg = getattr(response.generations[0][0], "message", None)
            usage_meta = getattr(gen_msg, "usage_metadata", None) or {}
            usage = {
                "prompt_tokens": usage_meta.get("input_tokens", 0),
                "completion_tokens": usage_meta.get("output_tokens", 0),
            }

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        model_name = (response.llm_output or {}).get("model_name", DEFAULT_MODEL_NAME)
        rates = PRICING.get(model_name, PRICING[DEFAULT_MODEL_NAME])
        cost = prompt_tokens * rates["input"] + completion_tokens * rates["output"]

        llm_cost_usd.add(cost, {"node": node, "model": model_name})
        llm_tokens.add(prompt_tokens, {"node": node, "direction": "prompt"})
        llm_tokens.add(completion_tokens, {"node": node, "direction": "completion"})
        if latency is not None:
            llm_latency_ms.record(latency, {"node": node})

        if self.CAPTURE_LLM_CONTENT:
            # A real span per LLM call, not just per node -- this is what
            # actually answers "what did retry attempt 2 see and say that
            # attempt 1 didn't."
            #
            # Sets BOTH OpenInference attributes (openinference.*, llm.*,
            # input.value/output.value) AND the OTel GenAI ones (gen_ai.*)
            # on the same span, deliberately, not redundantly. OpenInference
            # is Arize/Phoenix's own stable, mature convention -- it's what
            # actually drives Phoenix's LLM-span UI (prompt/completion
            # rendering, token counts) today. gen_ai.* is the newer,
            # still-developing OTel-community standard; Arize's cloud
            # product normalizes it on ingestion, but self-hosted Phoenix
            # support for it isn't a given. Setting both costs nothing and
            # means this works correctly regardless of which Phoenix
            # deployment or future OTel-GenAI-aware backend receives it.
            completion_text = ""
            try:
                completion_text = response.generations[0][0].text
            except (IndexError, AttributeError):
                pass
            prompt_capped = prompt_text[: self._MAX_CONTENT_CHARS]
            completion_capped = completion_text[: self._MAX_CONTENT_CHARS]
            with tracer.start_as_current_span(f"llm.{node}", attributes={"trace_id": self.trace_id}) as span:
                # OpenInference (Phoenix's native rendering convention)
                span.set_attribute("openinference.span.kind", "LLM")
                span.set_attribute("input.value", prompt_capped)
                span.set_attribute("input.mime_type", "text/plain")
                span.set_attribute("output.value", completion_capped)
                span.set_attribute("output.mime_type", "text/plain")
                span.set_attribute("llm.model_name", model_name)
                span.set_attribute("llm.provider", "openai")
                span.set_attribute("llm.token_count.prompt", prompt_tokens)
                span.set_attribute("llm.token_count.completion", completion_tokens)
                span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)
                # OTel GenAI semantic conventions (kept for forward-compat)
                span.set_attribute("gen_ai.system", "openai")
                span.set_attribute("gen_ai.request.model", model_name)
                span.set_attribute("gen_ai.prompt", prompt_capped)
                span.set_attribute("gen_ai.completion", completion_capped)
                span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                # Router-specific, not part of either convention
                span.set_attribute("cost_usd", cost)
                if latency is not None:
                    span.set_attribute("latency_ms", latency)

        self.llm_calls.append(
            {
                "node": node, "model": model_name,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "cost_usd": round(cost, 6), "latency_ms": latency,
            }
        )

    def on_llm_error(self, error, *, run_id, metadata=None, **kw):
        node = (metadata or {}).get("langgraph_node", "unknown")
        log_decision("llm_error", trace_id=self.trace_id, node=node, error=str(error))

    # ---- summary for the caller (answer_question) ----
    def get_summary(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "llm_call_count": len(self.llm_calls),
            "total_prompt_tokens": sum(c["prompt_tokens"] for c in self.llm_calls),
            "total_completion_tokens": sum(c["completion_tokens"] for c in self.llm_calls),
            "total_cost_usd": round(sum(c["cost_usd"] for c in self.llm_calls), 6),
        }


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]