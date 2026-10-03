# Matches the local dev environment's confirmed Python version
# (.venv paths throughout showed python3.13 explicitly).
FROM python:3.13-slim

WORKDIR /app

# build-essential included defensively -- some packages in this
# dependency tree occasionally need to compile from source on a platform
# without a prebuilt wheel. Worth trimming later as a non-blocking
# image-size optimization.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch FIRST. Default Linux torch from PyPI declares NVIDIA CUDA
# libraries (cublas, cudnn, nccl, ...) as dependencies -- multiple GB of GPU
# code that is useless in a CPU-only container, and which caused a
# mid-download DNS failure on nvidia_cublas in a real build. They never
# showed up in requirements.txt because pip freeze ran on macOS, where torch
# has no such dependencies. The pip install -r step below sees torch already
# satisfied at this pinned version and skips it.
# --resume-retries lets pip resume an interrupted large download instead of
# failing the whole layer on one network blip.
RUN pip install --no-cache-dir --resume-retries 5 \
    torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Copied and installed separately from the rest of the source so Docker's
# layer cache can skip the (slow) dependency install when only application
# code changed, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir --resume-retries 5 -r requirements.txt

# Application code. data/ is deliberately NOT copied here -- docker-compose.yml
# mounts it as a volume (facts.sqlite, chroma_store/, concept_embeddings.pkl
# are read-mostly, but audit_log.jsonl and hitl_review_queue.jsonl are written
# at runtime and need to survive a container restart).
COPY orchestration/ ./orchestration/
COPY eval/ ./eval/
COPY ingestion/ ./ingestion/
COPY api/ ./api/
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]