"""Retrieval pipeline shared by the CLI and the web app.

Storage is passed in as a `search(query_vector, n) -> list[dict]` function, so
the same steps work over ChromaDB (CLI) and the numpy index (web).
Each result dict has: text, source, page (or None), score (cosine similarity).
"""

import math
import re
from dataclasses import dataclass
from typing import Callable

from . import prompt as P
from . import thinking
from .config import Settings
from .ollama import OllamaClient, OllamaError, normalize

# Bump when stored vectors change meaning (prefixes, chunking), so old indexes
# are detected and rebuilt instead of silently giving worse results.
# v1 = original RagMind (no prefixes); v2 = task prefixes + page metadata.
INDEX_FORMAT = 2

# Some embedding models are trained with task prefixes; using them noticeably
# improves retrieval. Keyed by model name without tag.
_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}

FALLBACK_CTX = 4096
EMBED_BATCH = 32


def _prefixes(embed_model: str) -> tuple[str, str]:
    return _PREFIXES.get(normalize(embed_model).split(":")[0], ("", ""))


def index_signature(settings: Settings) -> dict:
    return {"embed_model": normalize(settings.embed_model), "format": INDEX_FORMAT}


def signature_problem(stored: dict | None, settings: Settings) -> str | None:
    """Why an existing index can't be searched with the current settings, or None."""
    if not stored:
        return "it was built by an older version of RagMind"
    if stored.get("format") != INDEX_FORMAT:
        return "it was built by an older version of RagMind"
    if stored.get("embed_model") != normalize(settings.embed_model):
        return (f"it was built with the embedding model '{stored.get('embed_model')}', "
                f"but the current one is '{normalize(settings.embed_model)}'")
    return None


def embed_documents(client: OllamaClient, settings: Settings, texts: list[str],
                    progress: Callable[[int, int], None] | None = None) -> list[list[float]]:
    prefix = _prefixes(settings.embed_model)[0]
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [prefix + t for t in texts[i:i + EMBED_BATCH]]
        vectors.extend(client.embed(settings.embed_model, batch))
        if progress:
            progress(len(vectors), len(texts))
    return vectors


def embed_query(client: OllamaClient, settings: Settings, text: str) -> list[float]:
    return client.embed(settings.embed_model, [_prefixes(settings.embed_model)[1] + text])[0]


def context_window(client: OllamaClient, settings: Settings, model: str) -> int:
    return min(client.context_length(model) or FALLBACK_CTX, settings.max_ctx)


_FOLLOWUP_WORDS = re.compile(
    r"\b(it|its|it's|they|them|their|this|that|these|those|he|she|his|her|there|"
    r"above|previous|earlier|former|latter|same|"
    r"(the )?(first|second|third|last|other|which) (one|point|item|step|part))\b", re.I)
_FOLLOWUP_START = re.compile(r"^\s*(and|but|also|so|then|what about|how about)\b", re.I)
_MORE = re.compile(r"\b(elaborate|expand on|more detail|in detail|explain further|go on|continue|"
                   r"tell me more|why is that|why not)\b", re.I)


def looks_like_followup(question: str) -> bool:
    """Does the question lean on the conversation? Only such questions are
    rewritten before searching and shown the earlier conversation; a complete
    question ("How does the round-table protocol resolve conflicts?") is
    searched as asked and answered from the documents alone."""
    q = question.strip()
    # "these papers", "both documents" point at the documents, not the conversation.
    bare = _CORPUS_REF.sub(" ", q)
    if _FOLLOWUP_WORDS.search(bare) or _FOLLOWUP_START.search(bare) or _MORE.search(bare):
        return True
    if is_broad(q):
        return False                      # "Summarise the key findings" stands on its own
    return len(q.split()) <= 5


def rewrite_followup(client: OllamaClient, model: str, question: str,
                     history: list[dict], num_ctx: int) -> str:
    """Turn a follow-up ("what about the second one?") into a standalone search query.
    Falls back to the original question if the model's reply looks unusable."""
    if not history or not looks_like_followup(question):
        return question
    try:
        reply = client.generate(model, P.build_rewrite_prompt(question, history),
                                {"temperature": 0, "num_predict": _reply_budget(client, model, 80),
                                 "num_ctx": num_ctx})
    except OllamaError:
        return question
    reply = thinking.strip(reply)
    lines = [l for l in reply.splitlines() if l.strip()]
    # A rewrite is one sentence. Several lines, or a list item / heading, means the
    # model answered or echoed the conversation instead (seen with ministral).
    if not lines or len(lines) > 3 or _LIST_OR_HEADING.match(lines[0]):
        return question
    line = _first_answer_line(reply)
    if not line or len(line) > max(300, 4 * len(question)):
        return question
    return line


_LIST_OR_HEADING = re.compile(r"^\s*(\d+[.)]\s|[-•#>|]|\*\s)")


_LEAD_INS = ("rewritten question:", "standalone question:", "question:")


def _first_answer_line(reply: str) -> str:
    """Models often add a lead-in ("Here is the rewritten question:") or quotes."""
    for raw in reply.splitlines():
        line = raw.strip().strip("*").strip()
        if not line or line.endswith(":"):
            continue                       # empty, or a lead-in line before the answer
        for lead in _LEAD_INS:
            if line.lower().startswith(lead):
                line = line[len(lead):].strip()
        return line.strip('"“”\'').strip()
    return ""


def grade_chunks(client: OllamaClient, model: str, question: str,
                 chunks: list[dict], num_ctx: int) -> list[dict]:
    """Keep only passages the model judges relevant. If every passage is rejected
    or grading fails, the original list is kept (better than answering blind)."""
    kept = []
    for ch in chunks:
        try:
            reply = client.generate(model, P.build_grade_prompt(question, ch["text"]),
                                    {"temperature": 0, "num_predict": _reply_budget(client, model, 3),
                                     "num_ctx": min(num_ctx, 2048)})
        except OllamaError:
            return chunks
        reply = thinking.strip(reply)
        if reply.strip().lstrip("*_\"' ").lower().startswith("yes"):
            kept.append(ch)
    return kept or chunks


def _reply_budget(client: OllamaClient, model: str, short_reply: int) -> int:
    """Reasoning models think before answering, so short replies need room.
    Thinking counts against num_predict, so without room the reply comes back
    empty (seen with gpt-oss, whose name gives no hint that it reasons)."""
    if thinking._REASONING_NAME.search(model):
        return 1024
    try:
        capabilities = client.show(model).get("capabilities") or []
    except OllamaError:
        capabilities = []
    return 1024 if "thinking" in capabilities else short_reply


def select(results: list[dict], top_k: int, min_score: float) -> list[dict]:
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    return [r for r in ranked if r["score"] >= min_score][:top_k]


@dataclass
class Prepared:
    prompt: str
    chunks: list[dict]          # passages actually placed in the prompt
    search_query: str           # what was searched (differs for rewritten follow-ups)
    num_ctx: int
    history_used: int
    broad: bool = False         # passages were drawn from every document


# ── broad questions ("summarise / compare the documents") ─────────────────────
# A plain similarity search for such a question returns whatever happens to be
# closest, often all from one document and often boilerplate (author roles,
# references). For these questions every document contributes its opening
# (title/abstract) plus its best matches, and boilerplate is skipped.

_DOC = r"(document|doc|paper|file|book|article|source|text|study|studie|report|pdf)"
_DOCS = rf"{_DOC}s\b"                      # plural: "the documents", "both papers"
_BROAD = [
    # summarise / compare ... documents
    re.compile(rf"\b(summar\w*|overview|outline|compare|comparison|contrast|differ\w*|similar\w*|"
               rf"in common)\b[^?!]*\b{_DOCS}", re.I),
    # documents ... compare / about
    re.compile(rf"\b{_DOCS}[^?!]*\b(compare|comparison|differ\w*|similar\w*|summar\w*|overview|about)\b", re.I),
    # both / each / all / these ... document(s)
    re.compile(rf"\b(both|each|all|every|these|those)\b(\s+\w+){{0,3}}\s+{_DOC}s?\b", re.I),
    # "what is this all about?", "what are they about?"
    re.compile(r"\bwhat (is|are) (it|they|this|these) (all )?about\b", re.I),
]
# "these papers", "both documents": a reference to the indexed documents.
_CORPUS_REF = re.compile(rf"\b(these|those|this|the|both|all|each|every)\b(\s+\w+){{0,3}}\s+{_DOC}s?\b", re.I)
_SHORT_SUMMARY = re.compile(r"\b(summar\w*|overview|tl;?dr)\b", re.I)
# Author-contribution (CRediT) lists: these terms rarely occur together elsewhere.
_CREDIT_ROLES = re.compile(
    r"\b(conceptuali[sz]ation|data curation|funding acquisition|formal analysis|"
    r"writing\s*[-–—]\s*original draft|writing\s*[-–—]\s*review|project administration)\b", re.I)
_BOILERPLATE_HEADINGS = re.compile(
    r"^\s*(references|bibliography|acknowledge?ments?|declaration of competing interest|"
    r"credit authorship contribution statement|author contributions?|data availability)\b", re.I)
MAX_BROAD_DOCS = 12
HEAD_SCAN = 16          # chunks searched for the abstract
TAIL_SCAN = 200         # chunks (from the end) searched for conclusion, appendix, references
MAX_SECTION = 8         # chunks taken from the abstract and from the conclusion

# Section headings, as they appear in whitespace-normalised chunk text.
_ABSTRACT = re.compile(r"\b(Abstract|ABSTRACT|A B S T R A C T)\b")
_INTRO = re.compile(r"(?:^|[\s.])(?:1\.?\s+|I\.\s+)?(Introduction|INTRODUCTION)\s+(?=[A-Z])")
_CONCLUSION = re.compile(
    r"(?:^|[\s.])(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?"
    r"(Conclusions?|CONCLUSIONS?|Concluding [Rr]emarks|CONCLUDING REMARKS|"
    r"Summary and [Cc]onclusions?|Discussion and [Cc]onclusions?)"
    # "6. Conclusion, limitations and future work"
    r"(?:[,:]?\s+(?:and|limitations|future|work|research|directions|outlook|perspectives|"
    r"remarks|lessons|learned)\b,?)*"
    r"\s+(?=[A-Z])")
_APPENDIX = re.compile(r"\b(Appendix|APPENDIX)\s+[A-Z]\.?\s+(?=[A-Z][a-z])")
_REFERENCES = re.compile(r"(?:^|\s)(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s+(?=\[\s?1\s?\]|1\.\s)")
_ASKS_ABOUT_REFERENCES = re.compile(r"\b(reference|cite[sd]?|citation|bibliograph)", re.I)
_FRONT_MATTER = re.compile(
    r"@|\buniversity\b|\bdepartment\b|\binstitute\b|\breceived\b|\baccepted\b|©|"
    r"\bdoi\b|journal homepage|\bissn\b|corresponding author|\bkeywords?\b|A R T I C L E", re.I)

_LAYOUTS: dict = {}


def is_broad(question: str) -> bool:
    """Does the question ask about the documents as a whole (several of them)?"""
    q = question.strip()
    if any(p.search(q) for p in _BROAD):
        return True
    # A bare "summarise" / "give me an overview" with no specific topic.
    return bool(_SHORT_SUMMARY.search(q)) and len(q.split()) <= 6


def is_boilerplate(text: str) -> bool:
    if _BOILERPLATE_HEADINGS.search(text):
        return True
    roles = {m.group(1).lower().split()[0] for m in _CREDIT_ROLES.finditer(text)}
    return len(roles) >= 2


def is_front_matter(text: str) -> bool:
    """Title-page material: authors, affiliations, e-mails, journal details."""
    return len({m.group(0).lower() for m in _FRONT_MATTER.finditer(text)}) >= 3


def doc_layout(corpus, key) -> dict:
    """Where a document's abstract, conclusion, appendices and reference list are.
    Positions are chunk ids; computed once per document version."""
    n = corpus.chunk_count(key)
    head = corpus.chunk_range(key, 0, min(HEAD_SCAN, n)) if n > 0 else []
    cache_key = (key, n, head[0]["text"][:80] if head else "")
    if cache_key in _LAYOUTS:
        return _LAYOUTS[cache_key]
    layout = {"abstract": [], "conclusion": [], "appendix_start": None, "references_start": None}
    if n <= 0:
        return layout

    # Abstract: from its heading up to (and including) the chunk where the
    # introduction starts, so the results sentences at its end are included.
    for i, ch in enumerate(head):
        if _ABSTRACT.search(ch["text"]):
            for c in head[i:i + MAX_SECTION]:
                layout["abstract"].append(c)
                if _INTRO.search(c["text"]):
                    break
            break
    if not layout["abstract"]:                        # no heading: first non-title-page chunks
        layout["abstract"] = [c for c in head[:6] if not is_front_matter(c["text"])][:3]

    tail_start = max(len(head), n - TAIL_SCAN)
    tail = corpus.chunk_range(key, tail_start, n) if tail_start < n else []
    body = [c for c in head + tail if (c.get("chunk_id") or 0) >= n * 0.4]
    for c in body:
        cid = c.get("chunk_id") or 0
        if layout["references_start"] is None and cid >= n * 0.5 and _REFERENCES.search(c["text"]):
            layout["references_start"] = cid
    conclusion_at = next((i for i, c in enumerate(body) if _CONCLUSION.search(c["text"])), None)
    if conclusion_at is not None:
        start_id = body[conclusion_at].get("chunk_id") or 0
        for c in body[conclusion_at:]:
            cid = c.get("chunk_id") or 0
            if cid > start_id and _APPENDIX.search(c["text"]):
                layout["appendix_start"] = cid
                break
        for c in body[conclusion_at:conclusion_at + MAX_SECTION]:
            cid = c.get("chunk_id") or 0
            if (is_boilerplate(c["text"]) or _APPENDIX.search(c["text"]) or
                    (layout["references_start"] is not None and cid >= layout["references_start"])):
                break
            layout["conclusion"].append(c)
    if layout["appendix_start"] is None:
        layout["appendix_start"] = next(
            (c.get("chunk_id") for c in body if (c.get("chunk_id") or 0) >= n * 0.5
             and _APPENDIX.search(c["text"])), None)
    _LAYOUTS[cache_key] = layout
    return layout


def in_back_matter(layout: dict, chunk_id, include_appendix: bool = True) -> bool:
    """Is this chunk in the reference list (or, optionally, an appendix)?"""
    if chunk_id is None:
        return False
    if layout["references_start"] is not None and chunk_id >= layout["references_start"]:
        return True
    return (include_appendix and layout["appendix_start"] is not None
            and chunk_id >= layout["appendix_start"])


def summary_chunks(corpus, key) -> list[dict]:
    """A document's abstract and conclusion: where authors state their findings."""
    layout = doc_layout(corpus, key)
    out = []
    for section in ("abstract", "conclusion"):
        for ch in layout[section]:
            out.append(dict(ch, section=section))
    return out


def broad_chunks(corpus, vector, settings: Settings, keys: list[str]) -> list[dict]:
    """Abstract, conclusion and best matches from each document, interleaved so
    that a tight prompt budget still keeps something from every document."""
    keys = keys[:MAX_BROAD_DOCS]
    matches = max(2, math.ceil(settings.top_k / len(keys)))
    groups = []
    for key in keys:
        picked, seen = [], set()
        layout = doc_layout(corpus, key)
        summary = summary_chunks(corpus, key)
        found = [h for h in corpus.search(vector, matches + 10, key)
                 if not in_back_matter(layout, h.get("chunk_id"))]
        for ch in summary + found:
            if ch["text"] in seen or is_boilerplate(ch["text"]):
                continue
            if ch.get("section") != "abstract" and is_front_matter(ch["text"]):
                continue
            seen.add(ch["text"])
            picked.append(ch)
            if len(picked) >= len(summary) + matches:
                break
        groups.append(picked)
    interleaved = []
    for i in range(max((len(g) for g in groups), default=0)):
        interleaved += [g[i] for g in groups if i < len(g)]
    return interleaved


def merge_overlapping(first: str, second: str) -> str:
    """Join two consecutive chunks, dropping the text they share (chunks overlap)."""
    for k in range(min(len(second), 300), 15, -1):
        if first.endswith(second[:k]):
            return first + second[k:]
    return first + " " + second


def with_following_text(corpus, chunks: list[dict]) -> list[dict]:
    """Extend each passage with the chunk after it. Chunks are short, so a passage
    often stops just before the detail that answers the question ("…we adopt
    three datasets." with the names in the next chunk)."""
    texts = {c["text"] for c in chunks}
    out = []
    for c in chunks:
        key, cid = c.get("key"), c.get("chunk_id")
        if key is not None and cid is not None:
            nxt = corpus.chunk_range(key, cid + 1, cid + 2)
            if nxt and nxt[0]["text"] not in texts and not is_boilerplate(nxt[0]["text"]) \
                    and not in_back_matter(doc_layout(corpus, key), cid + 1, include_appendix=False):
                c = dict(c, text=merge_overlapping(c["text"], nxt[0]["text"]))
        out.append(c)
    return out


def prepare(client: OllamaClient, settings: Settings, model: str, question: str,
            history: list[dict], search: Callable[[list[float], int], list[dict]],
            corpus=None) -> Prepared:
    """`corpus`, when given, enables broad questions. It needs:
    keys() -> document keys; search(vector, n, key) -> results within one
    document; chunk_count(key) -> number of chunks; chunk_range(key, start, stop) ->
    chunks in document order (with chunk_id)."""
    num_ctx = context_window(client, settings, model)
    history = [h for h in (history or []) if h.get("q") and h.get("a")]
    # The conversation is used only for questions that depend on it. A complete
    # question is answered from the documents alone, so claims from earlier
    # answers can't leak in as if they were facts.
    if not looks_like_followup(question):
        history = []
    search_query = question
    if history and settings.rewrite_followups and not is_broad(question):
        search_query = rewrite_followup(client, model, question, history, num_ctx)
    vector = embed_query(client, settings, search_query)

    keys = corpus.keys() if corpus is not None else []
    if len(keys) >= 2 and (is_broad(question) or is_broad(search_query)):
        chunks = broad_chunks(corpus, vector, settings, keys)
        names = list(dict.fromkeys(ch["source"] for ch in chunks))
        note = (f"The passages come from {len(names)} documents: " + "; ".join(names) +
                ". Passages marked 'abstract' or 'conclusion' state each document's own findings. "
                "Cover each document separately, then compare them if asked.")
        text, used, turns = P.build_prompt(question, chunks, history, num_ctx,
                                           settings.answer_reserve, settings.history_turns, note=note)
        return Prepared(text, used, search_query, num_ctx, turns, broad=True)

    results = search(vector, settings.top_k)
    if search_query != question:
        # A rewrite can drift from what was asked; search with the original
        # wording too and keep the best of both.
        merged = {}
        for r in results + search(embed_query(client, settings, question), settings.top_k):
            if r["text"] not in merged or r["score"] > merged[r["text"]]["score"]:
                merged[r["text"]] = r
        results = list(merged.values())
    if corpus is not None and not _ASKS_ABOUT_REFERENCES.search(question):
        # Reference lists match many questions on wording alone ("FWOS" came from one).
        results = [r for r in results if r.get("key") is None or not in_back_matter(
            doc_layout(corpus, r["key"]), r.get("chunk_id"), include_appendix=False)]
    chunks = select(results, settings.top_k, settings.min_score)
    if corpus is not None:
        chunks = with_following_text(corpus, chunks)
    if settings.grade_chunks and chunks:
        chunks = grade_chunks(client, model, search_query, chunks, num_ctx)
    text, used, turns = P.build_prompt(question, chunks, history, num_ctx,
                                       settings.answer_reserve, settings.history_turns)
    return Prepared(text, used, search_query, num_ctx, turns)


def generation_options(num_ctx: int) -> dict:
    return {"temperature": 0.1, "num_ctx": num_ctx}
