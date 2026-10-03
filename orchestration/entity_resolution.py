import re

COMPANY_NAMES = {
    "aapl": ["apple"], "jpm": ["jpmorgan", "jp morgan", "chase"],
    "xom": ["exxon", "exxonmobil", "exxon mobil"], "pfe": ["pfizer"],
    "wmt": ["walmart"], "pld": ["prologis"], "cat": ["caterpillar"],
    "met": ["metlife"], "ms": ["morgan stanley"], "tsla": ["tesla"],
}

def is_garbage_input(question:str)->bool:
    stripped=question.strip()
    if len(stripped)<3:
        return True
    if not re.search(r"[a-zA-Z]", stripped):
        return True
    words = stripped.split()
    if len(words) == 1 and re.search(r"\d", words[0]) and re.search(r"[a-zA-Z]", words[0]):
        return True
    return False


# Drop-in replacement for resolve_company in orchestration/entity_resolution.py
import re
import difflib

def resolve_company(question: str) -> str | None:
    lower = question.lower()
    matches = [
        ticker for ticker, names in COMPANY_NAMES.items()
        if any(re.search(rf"\b{re.escape(n)}\b", lower) for n in names)
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None  # genuinely ambiguous (multiple exact matches) -- fuzzy matching shouldn't touch this case

    # No exact match -- try a close-but-not-exact match on individual
    # words (catches "aaple" -> "apple"). Conservative cutoff (0.8) on
    # purpose: fuzzy company matching is a real place to be cautious,
    # since a too-loose match could silently answer about the wrong
    # company instead of asking for clarification -- a wrong answer is
    # worse than a correctly-blocked one here. Confirmed via testing:
    # exact matches, unrelated text, and genuinely ambiguous two-company
    # questions are all unaffected by this addition.
    words = [w.strip(",.?!") for w in lower.split()]
    all_names = [(name, ticker) for ticker, names in COMPANY_NAMES.items() for name in names]
    fuzzy_matches = set()
    for word in words:
        if len(word) < 4:
            continue  # too short for fuzzy matching to be reliable
        close = difflib.get_close_matches(word, [n for n, _ in all_names], n=1, cutoff=0.8)
        if close:
            matched_name = close[0]
            ticker = next(t for n, t in all_names if n == matched_name)
            fuzzy_matches.add(ticker)
    if len(fuzzy_matches) == 1:
        return fuzzy_matches.pop()
    return None

def is_out_of_domain(question:str,resolved_company:str|None)->bool:
    if resolved_company:
        return False
    off_topic_signals = ["weather", "recipe", "joke", "poem", "translate", "capital of"]
    return any(s in question.lower() for s in off_topic_signals)
_MORAL_JUDGMENT_PATTERNS = [
    r"\bwas\b.{0,80}\b(ethical|unethical)\b",
    r"\b(ethically|morally)\s+(justified|right|wrong|acceptable|defensible)\b",
    r"\bdo you (believe|think)\b.{0,80}\b(ethical|unethical|justified)\b",
]
 
_MOTIVE_SPECULATION_PATTERNS = [
    r"\bunderlying\b.{0,40}\b(objectives|motives|motivations|reasons)\b",
    r"\b(real|true|hidden|secret)\s+(reason|motive|intention)s?\b",
    r"\bdiscreetly\b",
    r"\bnot\s+(publicly\s+)?disclosed\b",
]
 
def classify_refusal_flavor(question:str)->str|None:
    lower=question.lower()
    for pattern in _MORAL_JUDGMENT_PATTERNS:
        if re.search(pattern,lower):
            return "moral_judgment"
        
    for pattern in _MOTIVE_SPECULATION_PATTERNS:
        if re.search(pattern,lower):
            return "motive_speculation"
    return None

REFUSAL_TEMPLATES = {
    "moral_judgment": (
        "I can't render a personal ethical judgment on this. I can share "
        "relevant factual details from the filing if that would help -- "
        "let me know what you're looking for specifically."
    ),
    "motive_speculation": (
        "I can't assert motives or intentions that aren't stated in the "
        "filing -- that would be speculation, not something the company "
        "has actually disclosed. I can share what the filing does say on "
        "this topic if that would help."
    ),
}