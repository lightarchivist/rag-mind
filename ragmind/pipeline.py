"""Retrieval pipeline shared by the CLI and the web app.

Storage is passed in as a `search(query_vector, n) -> list[dict]` function, so
the same steps work over ChromaDB (CLI) and the numpy index (web).
Each result dict has: text, source, page (or None), score (cosine similarity).
"""

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


def prepare(client: OllamaClient, settings: Settings, model: str, question: str,
            history: list[dict], search: Callable[[list[float], int], list[dict]]) -> Prepared:
    num_ctx = context_window(client, settings, model)
    history = [h for h in (history or []) if h.get("q") and h.get("a")]
    search_query = question
    if history and settings.rewrite_followups:
        search_query = rewrite_followup(client, model, question, history, num_ctx)
    results = search(embed_query(client, settings, search_query), settings.top_k)
    chunks = select(results, settings.top_k, settings.min_score)
    if settings.grade_chunks and chunks:
        chunks = grade_chunks(client, model, search_query, chunks, num_ctx)
    text, used, turns = P.build_prompt(question, chunks, history, num_ctx,
                                       settings.answer_reserve, settings.history_turns)
    return Prepared(text, used, search_query, num_ctx, turns)


def generation_options(num_ctx: int) -> dict:
    return {"temperature": 0.1, "num_ctx": num_ctx}
