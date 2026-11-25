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


def rewrite_followup(client: OllamaClient, model: str, question: str,
                     history: list[dict], num_ctx: int) -> str:
    """Turn a follow-up ("what about the second one?") into a standalone search query.
    Falls back to the original question if the model's reply looks unusable."""
    if not history:
        return question
    try:
        reply = client.generate(model, P.build_rewrite_prompt(question, history),
                                {"temperature": 0, "num_predict": 80, "num_ctx": num_ctx})
    except OllamaError:
        return question
    line = _first_answer_line(reply)
    if not line or len(line) > max(300, 4 * len(question)):
        return question
    return line


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
                                    {"temperature": 0, "num_predict": 3, "num_ctx": min(num_ctx, 2048)})
        except OllamaError:
            return chunks
        if reply.strip().lstrip("*_\"' ").lower().startswith("yes"):
            kept.append(ch)
    return kept or chunks


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
_SHORT_SUMMARY = re.compile(r"\b(summar\w*|overview|tl;?dr)\b", re.I)
# Author-contribution (CRediT) lists: these terms rarely occur together elsewhere.
_CREDIT_ROLES = re.compile(
    r"\b(conceptuali[sz]ation|data curation|funding acquisition|formal analysis|"
    r"writing\s*[-–—]\s*original draft|writing\s*[-–—]\s*review|project administration)\b", re.I)
_BOILERPLATE_HEADINGS = re.compile(
    r"^\s*(references|bibliography|acknowledge?ments?|declaration of competing interest|"
    r"credit authorship contribution statement|author contributions?|data availability)\b", re.I)
MAX_BROAD_DOCS = 12
OPENING_CHUNKS = 2


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


def broad_chunks(corpus, vector, settings: Settings, keys: list[str]) -> list[dict]:
    """Opening + best matches from each document, interleaved so that a tight
    prompt budget still keeps something from every document."""
    keys = keys[:MAX_BROAD_DOCS]
    per_doc = max(3, math.ceil(settings.top_k * 1.5 / len(keys)))
    groups = []
    for key in keys:
        picked, seen = [], set()
        for ch in corpus.opening(key, OPENING_CHUNKS) + corpus.search(vector, per_doc + 6, key):
            if ch["text"] in seen or is_boilerplate(ch["text"]):
                continue
            seen.add(ch["text"])
            picked.append(ch)
            if len(picked) >= per_doc:
                break
        groups.append(picked)
    interleaved = []
    for i in range(max((len(g) for g in groups), default=0)):
        interleaved += [g[i] for g in groups if i < len(g)]
    return interleaved


def prepare(client: OllamaClient, settings: Settings, model: str, question: str,
            history: list[dict], search: Callable[[list[float], int], list[dict]],
            corpus=None) -> Prepared:
    """`corpus`, when given, enables broad questions. It needs:
    keys() -> document keys; search(vector, n, key) -> results within one
    document; opening(key, n) -> the document's first n chunks."""
    num_ctx = context_window(client, settings, model)
    history = [h for h in (history or []) if h.get("q") and h.get("a")]
    search_query = question
    if history and settings.rewrite_followups:
        search_query = rewrite_followup(client, model, question, history, num_ctx)
    vector = embed_query(client, settings, search_query)

    keys = corpus.keys() if corpus is not None else []
    if len(keys) >= 2 and (is_broad(question) or is_broad(search_query)):
        chunks = broad_chunks(corpus, vector, settings, keys)
        names = list(dict.fromkeys(ch["source"] for ch in chunks))
        note = (f"The passages come from {len(names)} documents: " + "; ".join(names) +
                ". Cover each document separately, then compare them if asked.")
        text, used, turns = P.build_prompt(question, chunks, history, num_ctx,
                                           settings.answer_reserve, settings.history_turns, note=note)
        return Prepared(text, used, search_query, num_ctx, turns, broad=True)

    results = search(vector, settings.top_k)
    chunks = select(results, settings.top_k, settings.min_score)
    if settings.grade_chunks and chunks:
        chunks = grade_chunks(client, model, search_query, chunks, num_ctx)
    text, used, turns = P.build_prompt(question, chunks, history, num_ctx,
                                       settings.answer_reserve, settings.history_turns)
    return Prepared(text, used, search_query, num_ctx, turns)


def generation_options(num_ctx: int) -> dict:
    return {"temperature": 0.1, "num_ctx": num_ctx}
