import re
from typing import Literal, Optional
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI

_model=ChatOpenAI(model="gpt-4o-mini",temperature=0.0)

class QueryAnalysis(BaseModel):
    rewritten_question:str=Field(description="Cleaned up, complete, retrievable version of the quesiton")
    sub_questions:list[str]=Field(description="Split into independent sub-questions if genuinely multi-part; otherwise a single-item list with the rewritten question")
    needs_hyde:bool=Field(description="True if the question is abstract/reasoning-style (why/how/relate) rather than a direct literal lookup")
    refusal_flavors:list[Literal["moral_judgment","motive_speculation","none"]]=Field(
        description="Per sub-question, in the same order as sub_questions. "
        "'moral_judgment' if that sub-question asks you to render a personal "
        "ethical/moral verdict (e.g. 'was X ethical', 'do you believe X was justified'). "
        "'motive_speculation' if it asks you to assert undisclosed motives or "
        "intentions the filing itself never states (e.g. 'what were the real/hidden "
        "reasons', 'what objectives were not publicly disclosed'). "
        "'none' for a normal, directly answerable factual or narrative question."
    )

_structured_model=_model.with_structured_output(QueryAnalysis)

def is_trivially_simple(question:str)->bool:
    # Reverted to the original signature -- context resolution ("how
    # about their gross margin?" -> a fully standalone question) is now
    # condense_question's dedicated job, run BEFORE this ever sees the
    # question. By the time analyze_query/is_trivially_simple runs, the
    # question is already standalone, so the plain word-count heuristic
    # is valid again and doesn't need a history-aware carve-out.
    word_count=len(question.split())
    has_multiple_marks=question.count("?")>1
    has_conjuction_join=bool(re.search(r"\b(and|also)\b.*\b(what|how|why)\b", question, re.IGNORECASE))
    return word_count>4 and word_count<15 and not has_multiple_marks and not has_conjuction_join


def _format_history(conversation_history: Optional[list[dict]], max_turns: int = 10) -> str:
    """
    conversation_history: list of {"question": str, "answer": str,
    "company": str} dicts, oldest first. Only the most recent max_turns
    are shown -- older turns add prompt cost without much benefit, and
    answers are truncated since the model needs enough to resolve a
    reference, not the full text.

    Widened from 3 to 10 (confirmed necessary in production): with a
    3-turn window, an ORDINAL reference like "what was that FIRST revenue
    number again?" asked past turn 3 silently resolved to the wrong
    number -- the model had a revenue figure visible in the window, just
    not the one the person meant, and answered confidently anyway with no
    sign anything was off. 10 turns is generous for this product's
    realistic session length (a short back-and-forth about a filing, not
    a long-running chat), so the fix is sized to the actual usage
    pattern, not padded further than that.

    Turns are numbered by their ABSOLUTE position in the full
    conversation (not renumbered 1..N for whatever's currently in the
    window), specifically so an ordinal reference like "the first
    question" or "two questions ago" has a real anchor to resolve
    against, rather than the model having to infer position purely from
    list order.
    """
    if not conversation_history:
        return ""
    recent = conversation_history[-max_turns:]
    start_index = len(conversation_history) - len(recent) + 1
    lines = ["Recent conversation, numbered by turn (use the turn numbers "
             "to resolve ordinal references like 'the first question' or "
             "'two questions ago', and use the content to resolve "
             "references like 'it', 'their', 'that', or an implied "
             "company/period/metric that isn't explicit in the CURRENT "
             "question):"]
    for offset, turn in enumerate(recent):
        turn_number = start_index + offset
        lines.append(f"Turn {turn_number} Q: {turn.get('question', '')}")
        answer_preview = (turn.get("answer") or "")[:300]
        lines.append(f"Turn {turn_number} A: {answer_preview}")
    return "\n".join(lines) + "\n\n"


def condense_question(question: str, conversation_history: Optional[list[dict]] = None) -> str:
    """
    Dedicated, single-purpose call: rewrite the latest user message into a
    fully standalone question using prior conversation turns, BEFORE
    anything else (company resolution, caching, routing) ever sees it.
    Kept separate from analyze_query's terminology/decomposition/
    refusal-flavor job on purpose -- isolates context-resolution failures
    to one place instead of entangling them with three other jobs in one
    prompt. No-ops (skips the LLM call) with no history to condense
    against, since the first turn of a conversation has nothing to
    resolve and paying for a call here would be pure waste.
    """
    if not conversation_history:
        return question
    history_block = _format_history(conversation_history)
    prompt = (
        f"{history_block}"
        f"Latest message: \"{question}\"\n\n"
        f"Rewrite the latest message as a complete, standalone question "
        f"that could be understood with NO other context -- resolve any "
        f"pronoun, implied company, implied time period, or implied topic "
        f"using the conversation above. If the latest message refers to a "
        f"specific turn by position (e.g. 'the first question I asked', "
        f"'two questions ago', 'that earlier number'), use the Turn "
        f"numbers above to identify exactly which turn is meant, rather "
        f"than guessing based on topic similarity alone. If the latest "
        f"message is ALREADY "
        f"fully standalone (names its own company/topic explicitly and "
        f"doesn't rely on the conversation above), return it UNCHANGED. "
        f"Return ONLY the rewritten question, nothing else -- no preamble, "
        f"no quotes, no explanation."
    )
    result = _model.invoke(prompt).content.strip()
    return result if result else question


def analyze_query(question:str,company:str)->QueryAnalysis:
    # No conversation_history param -- condense_question (below) already
    # resolved any conversational context into a standalone question
    # before this runs, as its own dedicated call. Keeping that job in
    # exactly one place, rather than also handling it here, avoids two
    # prompts doing overlapping work and makes a context-resolution
    # failure isolated and debuggable at the condense_question step
    # specifically, rather than entangled with rewriting/decomposition/
    # refusal-flavor classification here.
    if is_trivially_simple(question):
        return QueryAnalysis(rewritten_question=question,sub_questions=[question],needs_hyde=False,refusal_flavors=["none"])
    prompt=(
        f"The user is asking about {company}. Their question: \"{question}\"\n\n"
        f"Analyze it:\n"
        f"1. Rewrite it as a clean, complete, specific question if it's too "
        f"short, has typos, or uses casual synonyms (e.g. 'profit' -> 'net "
        f"income', 'made' -> 'revenue') that won't match SEC filing terminology.\n"
        f"2. If it genuinely asks multiple distinct things, split into "
        f"separate sub-questions. Otherwise return one.\n"
        f"3. Decide if this needs a hypothetical-document search aid (true "
        f"for abstract/reasoning questions like 'why' or 'how does X relate "
        f"to Y'; false for direct literal lookups).\n"
        f"4. For EACH sub-question, classify its refusal_flavor based on "
        f"what THIS ORIGINAL question is really asking -- not the rewritten "
        f"wording -- since rewriting toward clean filing terminology can "
        f"strip words like 'discreetly' or 'secretly' that signal a "
        f"sub-question needs to be refused rather than answered from the "
        f"filing. 'moral_judgment' for a personal ethical verdict request, "
        f"'motive_speculation' for a request to assert undisclosed motives "
        f"the filing never states, 'none' otherwise."
    )
    return _structured_model.invoke(prompt)

def generate_hyde_passage(question:str)->str:
    prompt=(
        f"Write a short, plausible 3-4 sentence passage as it might appear "
        f"in an SEC 10-K, that would answer: \"{question}\"\n\n"
        f"This is HYPOTHETICAL, for search purposes only."
    )
    return _model.invoke(prompt).content.strip()