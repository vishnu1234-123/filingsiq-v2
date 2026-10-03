import re

# Confirmed in production: "Tesla had two operating segments" was a
# semantically correct answer that failed this check, because the regex
# only ever matched digits -- "two" never became a candidate number to
# compare at all. Small spelled-out numbers (the range small enough to
# plausibly show up in a sentence like this) are normalized to digits
# before the digit-regex runs, so this stays a metric fix, not a change
# to what counts as a correct answer.
_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def _spell_out_to_digits(text: str) -> str:
    def replace(match):
        return _WORD_TO_NUM[match.group(0).lower()]
    pattern = r"\b(" + "|".join(_WORD_TO_NUM.keys()) + r")\b"
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


# Confirmed in production: "MetLife has authorized an additional $3.0
# billion" is the exact correct answer (3,000,000,000), but the plain
# digit regex extracted "3.0" and compared that against 3 billion --
# a guaranteed mismatch. The regex has no way to know "billion" is a
# multiplier. This searches for a magnitude word immediately after each
# extracted number and scales accordingly before comparing.
_MAGNITUDE = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000, "trillion": 1_000_000_000_000}


def verbatim_number_match(generated_answer:str,expected_value)->bool:
    if expected_value is None:
        return None
    normalized = _spell_out_to_digits(generated_answer)

    # Number optionally followed by a magnitude word, e.g. "$3.0 billion"
    pattern = r"([-+]?\$?[\d,]+\.?\d*)\s*(thousand|million|billion|trillion)?"
    for num_str, magnitude_word in re.findall(pattern, normalized, re.IGNORECASE):
        if not num_str:
            continue
        cleaned = num_str.replace("$", "").replace(",", "")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        if magnitude_word:
            val *= _MAGNITUDE[magnitude_word.lower()]
        if abs(val-expected_value)<0.01:
            return True
        if expected_value!=0 and abs(val-expected_value)/abs(expected_value)<0.001:
            return True
    return False

from langchain_openai import ChatOpenAI

_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def refusal_compliance(generated_answer: str, refusal_flavor: str = "") -> bool:
    """
    LLM-judge check: does this answer correctly decline to speculate
    about undisclosed motive or render a moral judgment, rather than
    matching a fixed keyword list (which was proven too narrow --
    real correct refusals like 'The data provided does not contain
    any information regarding...' don't match any hardcoded phrase).
    """
    prompt = (
        f"Answer: \"{generated_answer}\"\n\n"
        f"Does this answer correctly DECLINE to speculate about undisclosed "
        f"motive or render a moral/ethical judgment, staying within what's "
        f"actually stated in available data? Reply with only YES or NO."
    )
    result = _judge.invoke(prompt).content.strip().upper()
    return result.startswith("YES")

# eval/generation_metrics.py -- add this
def faithfulness_check(generated_answer: str, retrieved_context: str) -> bool:
    """
    RAGAS-style faithfulness: does every claim in the generated answer
    actually trace back to the retrieved context, or did the model add
    something from its own general knowledge? This is the prose
    equivalent of context_actually_supports_answer for numeric rows.
    """
    prompt = (
        f"Retrieved context:\n{retrieved_context}\n\n"
        f"Generated answer:\n{generated_answer}\n\n"
        f"Does EVERY factual claim in the generated answer appear supported "
        f"by the retrieved context above? Reply YES only if all claims are "
        f"grounded in the context. Reply NO if the answer includes anything "
        f"not stated in the context, even if it's true or plausible."
    )
    result = _judge.invoke(prompt).content.strip().upper()
    return result.startswith("YES")

# eval/generation_metrics.py -- corrected version
import re

def context_actually_supports_answer(retrieved_chunks: list[str], expected_value) -> bool:
    """
    Extracts actual numeric tokens from retrieved text and compares
    them numerically to the expected value -- avoids the bug where
    stripping trailing zeros from small integers (e.g. 2.0 -> "2")
    produces a single-character substring that matches almost anything.
    """
    combined = " ".join(retrieved_chunks)
    numbers_in_context = re.findall(r"[-+]?\d[\d,]*\.?\d*", combined)
    for n in numbers_in_context:
        cleaned = n.replace(",", "")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        if abs(val - expected_value) < 0.01:
            return True
        if expected_value != 0 and abs(val - expected_value) / abs(expected_value) < 0.001:
            return True
    return False

def faithfulness_score(generated_answer: str, retrieved_context: str, trace_id: str = "n/a") -> float:
    """
    Graded 0-1 faithfulness score. Logs the judge's raw reply on any
    parse failure so a genuine 0 (judge said "0") is distinguishable
    from a swallowed failure (judge said something non-numeric) --
    both currently collapse into the same silent 0.0 return.
    """
    prompt = (
        f"Retrieved context:\n{retrieved_context}\n\n"
        f"Generated answer:\n{generated_answer}\n\n"
        f"On a scale of 0-100, how well does every claim in the generated "
        f"answer trace back to the retrieved context? 100 = fully grounded, "
        f"0 = entirely unsupported. Reply with ONLY the number."
    )
    result = _judge.invoke(prompt).content.strip()
    try:
        return float(result) / 100
    except ValueError:
        from orchestration.telemetry import log_decision
        log_decision(
            "faithfulness_score_parse_failure", trace_id=trace_id,
            raw_judge_reply=result, context_len=len(retrieved_context),
            context_empty=len(retrieved_context.strip()) == 0,
        )
        return 0.0


def correctness_score(generated_answer: str, ground_truth: str) -> float:
    """
    Same pattern for correctness -- graded score against known-correct
    ground truth from your golden set.
    """
    prompt = (
        f"Generated answer: \"{generated_answer}\"\n\n"
        f"Known correct answer: \"{ground_truth}\"\n\n"
        f"On a scale of 0-100, how well does the generated answer convey "
        f"the same correct information as the known correct answer, even "
        f"if worded differently? Reply with ONLY the number."
    )
    result = _judge.invoke(prompt).content.strip()
    try:
        return float(result) / 100
    except ValueError:
        return 0.0