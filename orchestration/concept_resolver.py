import sqlite3
import os
import re

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
DB_PATH=os.path.join(_THIS_DIR,"..","data","facts.sqlite")

STOPWORDS = {"the", "did", "how", "much", "and", "was", "for", "in", "of", "to", "a", "did"}

# Bridges colloquial question vocabulary to XBRL taxonomy tag fragments.
# The original substring match only worked when a question happened to
# use taxonomy jargon verbatim ("EffectiveIncomeTaxRate") -- real
# questions say "R&D spend," "profit," "tax rate," none of which are
# literal substrings of the actual tags (ResearchAndDevelopmentExpense,
# NetIncomeLoss, EffectiveIncomeTaxRateContinuingOperations). This is a
# stopgap, not a permanent solution: it's a manually maintained dict that
# will need entries added every time a real question comes back empty.
# The more durable fix is to have the LLM (already in this call path,
# see run_sql's concept-pick step) generate likely taxonomy fragments
# from the question directly, rather than hand-curating synonyms --
# worth doing once this dict starts feeling like whack-a-mole.
ALIASES: dict[str, list[str]] = {
    "r&d": ["research", "development"],
    "rd": ["research", "development"],
    "spend": ["expense", "cost"],
    "spending": ["expense", "cost"],
    "spent": ["expense", "cost"],
    "revenue": ["revenue", "sales"],
    "sales": ["revenue", "sales"],
    "profit": ["netincome", "profit", "operatingincome"],
    "income": ["netincome", "income"],
    "earnings": ["netincome", "earnings"],
    "tax": ["tax"],
    "debt": ["debt", "longtermdebt", "liabilities"],
    "cash": ["cash", "cashandcashequivalents"],
    "assets": ["assets"],
    "liabilities": ["liabilities"],
    "shares": ["shares", "sharesoutstanding"],
}


def _search_terms(question: str) -> set[str]:
    """
    Every raw token gets checked against ALIASES first (so short but
    meaningful tokens like "r&d" aren't lost to the length filter below);
    tokens with no alias fall back to the original length/stopword rule.
    """
    raw_words = [w.strip(",.?").lower() for w in question.split()]
    terms: set[str] = set()
    for w in raw_words:
        if w in ALIASES:
            terms.update(ALIASES[w])
        elif len(w) > 3 and w not in STOPWORDS:
            terms.add(w)
    return terms


def has_gaap_vocabulary(question: str) -> bool:
    """
    True if the question uses vocabulary from the curated ALIASES table
    ("R&D", "profit", "spend", "sales", ...), i.e. words we have
    explicitly registered as colloquial names for GAAP line items.
    candidate_concepts itself is a substring search that reranks but
    never filters, so "found candidates" only means some common word
    appeared inside some concept name -- confirmed on real data to fire
    on narrative questions ("capital", "market", "financial"). This is
    the narrower signal the classifier override was actually built for.
    """
    raw_words = [w.strip(",.?").lower() for w in question.split()]
    return any(w in ALIASES for w in raw_words)


_FIGURE_REQUEST_PATTERNS = [
    re.compile(p) for p in (
        r"\bhow (much|many)\b",
        r"\bwhat percentage\b",
        r"\bwhat (is|was|are|were) (the|its|their) (total |aggregate |net )?(value|amount|number|balance)\b",
        r"\bwhat (are|were) the total\b",
        r"\bper share\b",
        r"\bfair value\b",
    )
]


def asks_for_specific_figure(question: str) -> bool:
    """
    Question WORDING that requests a reported figure ("how much", "what
    is the total amount of", "per share"), independent of vocabulary.
    Added because a vocabulary-only gate would break formally-worded SQL
    questions the classifier misses (jpm_sql_4, xom_sql_6, ms_sql_18...).
    A heuristic: it cannot tell a figure asked for from prose from one
    asked for from a table (pfe_corpaction_1 looks the same).
    """
    lower = question.lower()
    return any(p.search(lower) for p in _FIGURE_REQUEST_PATTERNS)


def looks_like_figure_request(question: str) -> bool:
    """
    Gate for the classifier override in dispatch_subquestions.
    candidate_concepts alone is not evidence a figure is being asked
    for: it is a substring search that reranks but never filters, and
    on the real golden set it returned candidates for 20 of 23
    narrative questions ("competitive position" matched concept names
    containing "market", "capital", etc.). Confirmed on the real golden
    set: gating the override on this function instead of "any candidate
    found" fixed 19 of those 20 wrongly-flipped rows (aapl_geo_1 is the
    one exception -- "net sales" is itself a real alias, so it still
    opens the gate) while still rescuing 11 of the 17 sql rows the
    classifier alone gets wrong, at the cost of no longer rescuing 6
    more that use neither GAAP vocabulary nor figure-request wording
    (jpm_sql_4, xom_sql_6, pld_sql_12, met_sql_16, ms_sql_18,
    tsla_sql_19) -- a real, measured tradeoff, not a free win.
    """
    return has_gaap_vocabulary(question) or asks_for_specific_figure(question)


def _split_concept_name(concept: str) -> str:
    """
    'us-gaap:OtherAssetsNoncurrent' -> 'other assets noncurrent'.
    Splitting CamelCase into words is what makes similarity ranking (TF-IDF
    here, a real sentence embedder in production) actually work -- the raw
    tag string reads as one token to most text-similarity methods otherwise.
    """
    name = concept.split(":")[-1]
    words = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", name)
    return " ".join(words).lower()


def _rerank_by_similarity(candidates: list[str], question: str, top_k: int = 20) -> list[str]:
    """
    Shortlists a large candidate set down to the top_k most relevant to
    the question before the LLM ever sees them. Confirmed empirically that
    naive substring matching on generic words ("other", "assets") can pull
    in dozens of irrelevant concepts while contributing nothing for the
    actually-distinguishing term ("non-current" never matches -- XBRL tags
    use camelCase "Noncurrent", not a literal hyphen). A long, noisy
    candidate list measurably hurts LLM pick accuracy even well within its
    context window (needle-in-haystack effect), separate from whether it
    technically fits.

    Primary method: mpnet sentence embeddings (same model already loaded
    for the Chroma vector store -- reusing it here costs no extra API
    money and only the cost of one more local forward pass, not an LLM
    call). This is what actually closes the "noncurrent" ~ "non-current"
    gap that TF-IDF's literal-token matching can't -- confirmed in testing
    that TF-IDF ranked the correct concept 6th out of 17 on exactly that
    mismatch. Concept embeddings are cached to disk (see
    _get_concept_embeddings) since the concept vocabulary is fixed and
    small -- no reason to re-embed ~1,500 concept names on every request.

    Falls back to TF-IDF only if sentence-transformers isn't installed or
    the model fails to load -- a dependency-robustness fallback, not a
    cost/latency tradeoff. If you've confirmed mpnet is always available
    in your deployment, this fallback branch will simply never trigger.
    """
    if len(candidates) <= top_k:
        return candidates

    try:
        return _rerank_with_mpnet(candidates, question, top_k)
    except Exception as exc:
        log_decision = _lazy_log_decision()
        if log_decision:
            log_decision("mpnet_rerank_failed", trace_id="n/a", error=str(exc), fallback="tfidf")
        try:
            return _rerank_with_tfidf(candidates, question, top_k)
        except Exception as exc2:
            # Both rerankers failed -- this should never crash the pipeline
            # over what is, at worst, an accuracy regression (LLM sees an
            # unranked, possibly-long list instead of a shortlist). Degrade
            # to the raw candidates rather than propagate. This is also
            # exactly how this bug surfaced in production: TF-IDF's own
            # failure (NameError on 're') was uncaught here and crashed
            # dispatch_subquestions entirely instead of just skipping the
            # ranking step.
            if log_decision:
                log_decision("tfidf_rerank_also_failed", trace_id="n/a", error=str(exc2), fallback="unranked")
            return candidates[:top_k]


def _lazy_log_decision():
    try:
        from orchestration.telemetry import log_decision
        return log_decision
    except Exception:
        return None


_MPNET_MODEL = None
_CONCEPT_EMBEDDING_CACHE_PATH = os.path.join(_THIS_DIR, "..", "data", "concept_embeddings.pkl")


def _get_mpnet_model():
    global _MPNET_MODEL
    if _MPNET_MODEL is None:
        from sentence_transformers import SentenceTransformer
        # Same model as the "filing_chunks_mpnet" Chroma collection -- if
        # your app already holds a loaded instance elsewhere, wire that in
        # here instead of loading a second copy into memory.
        _MPNET_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _MPNET_MODEL


def _get_concept_embeddings(concepts: list[str]) -> dict:
    """
    Returns {concept: embedding} for the given concepts, using an on-disk
    cache so the fixed, small concept vocabulary (~1,500 distinct tags
    across your ingested companies) only ever gets embedded once, not on
    every request. New concepts (e.g. after ingesting another company)
    get embedded and added to the cache automatically.
    """
    import pickle
    import numpy as np

    cache: dict = {}
    if os.path.exists(_CONCEPT_EMBEDDING_CACHE_PATH):
        with open(_CONCEPT_EMBEDDING_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

    missing = [c for c in concepts if c not in cache]
    if missing:
        model = _get_mpnet_model()
        texts = [_split_concept_name(c) for c in missing]
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        for c, v in zip(missing, vectors):
            cache[c] = v
        os.makedirs(os.path.dirname(_CONCEPT_EMBEDDING_CACHE_PATH), exist_ok=True)
        with open(_CONCEPT_EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    return {c: cache[c] for c in concepts}


def _rerank_with_mpnet(candidates: list[str], question: str, top_k: int) -> list[str]:
    """
    Uses sklearn's cosine_similarity on stacked embedding matrices rather
    than a hand-rolled per-pair dot-product loop -- the hand-rolled version
    crashed with "only 0-dimensional arrays can be converted to Python
    scalars" in production, because model.encode()'s output shape isn't
    guaranteed to be the flat (dim,) vector a naive float(np.dot(...))
    assumes across all sentence-transformers versions. Stacking into
    proper 2D matrices and letting sklearn handle the shapes is the same
    pattern already used in _rerank_with_tfidf below, so both paths are
    now consistent.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    model = _get_mpnet_model()
    concept_embeddings = _get_concept_embeddings(candidates)
    concept_matrix = np.stack([np.asarray(concept_embeddings[c]).reshape(-1) for c in candidates])
    question_vec = np.asarray(
        model.encode([question], convert_to_numpy=True, show_progress_bar=False)[0]
    ).reshape(1, -1)

    sims = cosine_similarity(question_vec, concept_matrix).flatten()
    ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]


def _rerank_with_tfidf(candidates: list[str], question: str, top_k: int) -> list[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return candidates[:top_k]  # last-resort degrade if sklearn isn't installed either

    docs = [_split_concept_name(c) for c in candidates] + [question.lower()]
    tfidf = TfidfVectorizer().fit_transform(docs)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]


def candidate_concepts(source_file_pattern:str,question:str)->list[str]:
    words = _search_terms(question)
    conn=sqlite3.connect(DB_PATH)
    found=set()
    for w in words:
        cur=conn.execute(
            "SELECT DISTINCT concept FROM facts WHERE source_file LIKE ? AND LOWER(concept) LIKE ?",
            (f"%{source_file_pattern}%",f"%{w}%"),
        )
        found.update(row[0] for row in cur.fetchall())
    conn.close()
    return _rerank_by_similarity(sorted(found), question)

def list_matching_concepts(source_file_pattern:str,keyword:str)->list[str]:
    conn=sqlite3.connect(DB_PATH)
    cur=conn.execute(
        "SELECT DISTINCT concept FROM facts "
        "WHERE source_file LIKE ? AND concept LIKE ?",
        (f"%{source_file_pattern}%",f"%{keyword}%"),
    )
    concepts=[row[0] for row in cur.fetchall()]
    conn.close()
    return concepts

def resolve_concept(source_file_pattern:str,must_include:list[str],must_exclude:list[str]=None,)->list[tuple]:
    must_exclude=must_exclude or []
    conn=sqlite3.connect(DB_PATH)
    cur=conn.execute(
        "SELECT concept,value,period_end,instant_date,segments_json "
        "FROM facts WHERE source_file LIKE ?",
        (f"%{source_file_pattern}%",),
    )

    rows=cur.fetchall()
    conn.close()

    matches=[]
    for r in rows:
        concept=r[0]
        if all(term in concept for term in must_include) and not any(term in concept for term in must_exclude):
            matches.append(r)
    return matches