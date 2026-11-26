"""Retrieval pipeline shared by the CLI and the web app.

Storage is passed in as a `search(query_vector, n) -> list[dict]` function, so
the same steps work over ChromaDB (CLI) and the numpy index (web).
Each result dict has: text, source, page (or None), score (cosine similarity).
"""

import math
import os
import re
from dataclasses import dataclass
from typing import Callable

from . import prompt as P
from . import thinking
from .config import Settings
from .ollama import OllamaClient, OllamaError, normalize

# Bump when stored vectors change meaning (prefixes, chunking), so old indexes
# are detected and rebuilt instead of silently giving worse results.
# v1 = original RagMind (no prefixes); v2 = task prefixes + page metadata;
# v3 = document titles and section headings, embedded with each chunk.
INDEX_FORMAT = 3

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


MAX_TITLE = 120


def embedding_text(text: str, title: str = "", heading: str = "") -> str:
    """What is embedded for a chunk: the document's title and the chunk's section,
    then the chunk. A passage that says "the proposed method" is then also about
    the paper and the section it is in. The stored text stays the chunk alone."""
    context = " | ".join(p for p in (title[:MAX_TITLE].strip(), heading.strip()) if p)
    return f"{context}\n{text}" if context else text


def embed_query(client: OllamaClient, settings: Settings, text: str) -> list[float]:
    return client.embed(settings.embed_model, [_prefixes(settings.embed_model)[1] + text])[0]


def context_window(client: OllamaClient, settings: Settings, model: str) -> int:
    return min(client.context_length(model) or FALLBACK_CTX, settings.max_ctx)


_FOLLOWUP_WORDS = re.compile(
    r"\b(it|its|it's|they|them|their|this|these|those|he|she|his|her|there|"
    r"above|previous|earlier|former|latter|same|"
    r"(the )?(first|second|third|last|other|which) (one|point|item|step|part))\b", re.I)
# "that" points back at the conversation in "explain that", "is that true?",
# "how does that work?", "that one", "that method"; but not when it joins a
# clause: "What does TI-LFA provide that classic LFA does not?"
_THAT_REFERS = re.compile(
    r"\bthat\s*(?:[?.!,;:]|$)"
    r"|\b(?:about|of|on|for|with|from|like|regarding|explain|describe|clarify|is|was|are|were|"
    r"does|did|do|can|could|would|will|should|why|how)\s+that\b"
    r"|\bthat\s+(?:one|ones|step|point|part|item|approach|method|technique|model|paper|document|"
    r"book|chapter|section|table|figure|example|answer|result|protocol|concept|feature|option|"
    r"case|number|value|idea|solution|tool|list|command|configuration|setting|claim)s?\b", re.I)
_FOLLOWUP_START = re.compile(r"^\s*(and|but|also|so|then|what about|how about)\b", re.I)
_MORE = re.compile(r"\b(elaborate|expand on|more detail|in detail|explain further|go on|continue|"
                   r"tell me more|why is that|why not)\b", re.I)


# A pronoun in a second clause can point back at the first clause of the same
# question: "What are the three signals in the framework, and how are they combined?"
_CLAUSE_BREAK = re.compile(r"[,;]|\b(?:and|or|but)\b", re.I)
_PRONOUN = re.compile(r"^(it|its|it's|they|them|their|this|these|those|he|she|his|her)$", re.I)
MIN_FIRST_CLAUSE = 5    # words a first clause needs before a later pronoun can refer to it


def _refers_within(text: str, match) -> bool:
    """Is this pronoun in a later clause, after a first clause long enough to
    name what it refers to?"""
    if not _PRONOUN.match(match.group(0)):
        return False
    breaks = list(_CLAUSE_BREAK.finditer(text, 0, match.start()))
    if not breaks:
        return False
    first = text[:breaks[0].start()]
    return len(first.split()) >= MIN_FIRST_CLAUSE and not _FOLLOWUP_WORDS.search(first)


def looks_like_followup(question: str, one_document: bool = False) -> bool:
    """Does the question lean on the conversation? Only such questions are
    rewritten before searching and shown the earlier conversation; a complete
    question ("How does the round-table protocol resolve conflicts?") is
    searched as asked and answered from the documents alone. With the chat
    limited to `one_document`, a short question ("What are the main
    limitations?") is about that document, not about the last answer."""
    q = question.strip()
    # "these papers", "both documents" point at the documents, not the conversation.
    bare = _CORPUS_REF.sub(" ", q)
    pointing = [m for m in _FOLLOWUP_WORDS.finditer(bare) if not _refers_within(bare, m)]
    if (pointing or _THAT_REFERS.search(bare) or _FOLLOWUP_START.search(bare)
            or _MORE.search(bare)):
        return True
    if is_broad(q) or one_document:
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
    return _keep_descriptions(question, line)


# "the smallest model", "the best configuration": the user describes something
# without naming it. Rewrites sometimes name it anyway, from the previous answer
# ("the smallest model, Llama 3.1 8B,") and send the search to the wrong place.
_DESCRIPTION = re.compile(r"\b(\w+est|best|worst|most|least|first|last|other|previous|next)\b", re.I)


def _keep_descriptions(question: str, rewrite: str) -> str:
    words = {w.lower() for w in _DESCRIPTION.findall(question)}
    for word in words:
        if not re.search(rf"\b{re.escape(word)}\b", rewrite, re.I):
            return question                      # the description was replaced or dropped
        # Drop a name put next to it: "smallest model, Llama 3.1 8B," / "smallest model (X)"
        rewrite = re.sub(rf"(\b{re.escape(word)}\s+\w+)\s*(?:,[^,?]{{1,60}},|\([^)]{{1,60}}\)|[-–—:]\s*[A-Z][^,?]{{0,60}}(?=[,?]))",
                         r"\1", rewrite, flags=re.I)
    return re.sub(r"\s{2,}", " ", rewrite).strip()


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


MAX_KEYWORD_ONLY = 2    # passages that keyword search may add to the similarity results


def fuse(similar: list[dict], keyword: list[dict], top_k: int, min_score: float) -> list[dict]:
    """Combine similarity and keyword results. Similarity results stay the basis:
    passages found by both searches come first, then the best similarity-only
    ones, and keyword search adds at most MAX_KEYWORD_ONLY passages that the
    similarity search missed (more only if similarity found too few)."""
    ranked = [r for r in sorted(similar, key=lambda r: r["score"], reverse=True)
              if r["score"] >= min_score]
    found = {r["text"] for r in ranked}
    words = {r["text"] for r in keyword}
    both = [r for r in ranked if r["text"] in words]
    similar_only = [r for r in ranked if r["text"] not in words]
    keyword_only = [r for r in keyword if r["text"] not in found]
    chosen = both[:top_k]
    room = top_k - len(chosen)
    extra = min(MAX_KEYWORD_ONLY, len(keyword_only), room)
    chosen += similar_only[:room - extra]
    chosen += keyword_only[:top_k - len(chosen)]
    return chosen


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
SHORT_DOC = 150         # chunks (about 25 pages): longer documents without an abstract are books

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
# Publisher pages in books: companion websites, access codes, offers.
_PUBLISHER = re.compile(
    r"companion web ?site|access code|pearson test prep|practice test software|"
    r"register (?:your|this) (?:book|product)|\bisbn\b|coupon|discount code|special offers?|"
    r"exclusive offers?|video training|informit|study planner|flash ?cards|premium edition|"
    r"bonus (?:content|material)|newsletter", re.I)
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
    if len(roles) >= 2:
        return True
    return len({m.group(0).lower() for m in _PUBLISHER.finditer(text)}) >= 2


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
    layout = {"abstract": [], "conclusion": [], "appendix_start": None, "references_start": None,
              "book": False}
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
    # No abstract heading: a short document (notes, a report) is represented by its
    # opening; a long one (a book) by nothing here, since its first pages are
    # copyright notices, contents and publisher offers, not findings.
    paper_like = bool(layout["abstract"]) or n <= SHORT_DOC
    if not layout["abstract"] and paper_like:
        layout["abstract"] = [c for c in head[:6] if not is_front_matter(c["text"])
                              and not is_boilerplate(c["text"])][:3]
    layout["book"] = not paper_like

    tail_start = max(len(head), n - TAIL_SCAN)
    tail = corpus.chunk_range(key, tail_start, n) if tail_start < n else []
    body = [c for c in head + tail if (c.get("chunk_id") or 0) >= n * 0.4]
    for c in body:
        cid = c.get("chunk_id") or 0
        if layout["references_start"] is None and cid >= n * 0.5 and _REFERENCES.search(c["text"]):
            layout["references_start"] = cid
    conclusion_at = None if layout["book"] else \
        next((i for i, c in enumerate(body) if _CONCLUSION.search(c["text"])), None)
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


def _overlap(first: str, second: str) -> int:
    """How many characters at the start of `second` repeat the end of `first`."""
    for k in range(min(len(second), 300), 15, -1):
        if first.endswith(second[:k]):
            return k
    return 0


def merge_overlapping(first: str, second: str) -> str:
    """Join two consecutive chunks, dropping the text they share (chunks overlap)."""
    k = _overlap(first, second)
    return first + second[k:] if k else first + " " + second


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


# ── context a passage needs to be read correctly ──────────────────────────────

MAX_LEAD = 350          # characters of the preceding text put before a passage
# Abbreviations whose full stop does not end a sentence ("in Ref. [12], the authors…").
_ABBREVIATIONS = {"ref", "refs", "fig", "figs", "eq", "eqs", "sec", "tab", "no", "vol", "pp",
                  "al", "e.g", "i.e", "cf", "vs", "etc", "approx", "resp", "dr", "prof", "st"}
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s+(?=[\"'(]?[A-Z0-9])")


def _sentence_starts(text: str) -> list[int]:
    starts = []
    for m in _SENTENCE_END.finditer(text):
        word = re.search(r"([\w.]+)[.!?][\"')\]]?$", text[:m.start() + 1])
        if word and word.group(1).lower().rstrip(".") in _ABBREVIATIONS:
            continue
        starts.append(m.end())
    return starts


def lead_in(previous: str, passage: str) -> str:
    """The sentence or two before a passage, so that "the authors" or "this method"
    at its start can be read correctly ("For instance, in Ref. [12], the authors…")."""
    before = previous[:len(previous) - _overlap(previous, passage)].rstrip()
    if not before:
        return ""
    window = before[-MAX_LEAD:]
    starts = [i for i in _sentence_starts(window) if i < len(window)]
    if starts:
        return window[starts[0]:].strip()
    if len(before) <= MAX_LEAD:
        return before.strip()
    cut = window.find(" ")
    return "…" + window[cut + 1:].strip() if cut >= 0 else ""


def with_preceding_text(corpus, chunks: list[dict]) -> list[dict]:
    """Put the end of the previous chunk before each passage."""
    texts = {c["text"] for c in chunks}
    out = []
    for c in chunks:
        key, cid = c.get("key"), c.get("chunk_id")
        if key is not None and cid:
            prev = corpus.chunk_range(key, cid - 1, cid)
            if prev and prev[0]["text"] not in texts and not is_boilerplate(prev[0]["text"]) \
                    and not is_front_matter(prev[0]["text"]):
                lead = lead_in(prev[0]["text"], c["text"])
                if lead:
                    c = dict(c, text=lead + " " + c["text"], lead=lead)
        out.append(c)
    return out


# Table captions. In extracted PDF text a table's rows run straight into the next
# table's caption, so a passage that ends one table and starts another reads as
# if the first table's numbers belonged to the second (Table 3's rows read as
# part of Table 4). Each such passage is labelled with its own table.
# "Table 4", "TABLE IV", and chapter-numbered "Table 5-3" / "Table 5.3" (books).
_CAPTION = re.compile(r"(?<![^\s.])(Table|TABLE)\s+(\d+(?:[-–.]\d+)?|[IVX]+)(?:\s*[.:]\s*|\s+)(?=[A-Z])")
_NOT_CAPTION_AFTER = {"in", "see", "of", "and", "to", "from", "with", "by", "per", "as", "on",
                      "under", "shown", "listed", "reported", "summarized", "summarised", "given",
                      "following", "next", "previous", "this", "that", "the"}
# A number in a table cell: 42, 3.15%, 1,024, (0.5). Not an IP address (10.1.1.1)
# or a time (00:01:02): command output is not a table.
_NUMBER = re.compile(r"^[(\[]?[-+±]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)%?[)\]]?[,;]?$")
TABLE_SCAN = 8          # chunks searched for the caption of the table a passage starts in
_PLACEMENTS: dict = {}


def captions(text: str) -> list[tuple[int, str]]:
    """(position, "Table 3: Performance comparison on …") for each table caption."""
    out = []
    for m in _CAPTION.finditer(text):
        word = re.search(r"(\w+)\W*$", text[:m.start()])
        if word and word.group(1).lower() in _NOT_CAPTION_AFTER:
            continue
        rest = text[m.end():m.end() + 200]
        title = re.match(r"(.{3,160}?[.!?])(?=\s|$)", rest)
        title = title.group(1) if title else rest[:120].rstrip() + "…"
        out.append((m.start(), f"{m.group(1).title()} {m.group(2)}: {title}"))
    return out


def has_table_rows(text: str, need: int = 6, window: int = 12) -> bool:
    """Does the text contain table rows (a stretch dense with numbers)?"""
    flags = [bool(_NUMBER.match(t)) for t in text.split()]
    return any(sum(flags[i:i + window]) >= need for i in range(max(1, len(flags) - window + 1)))


def caption_placement(corpus, key) -> str:
    """'above' or 'below': where this document prints table captions. Most journals
    put them above; a document votes the other way only on clear evidence."""
    n = corpus.chunk_count(key)
    if n <= 0:
        return "above"
    first = corpus.chunk_range(key, 0, 1)
    cache_key = (key, n, first[0]["text"][:80] if first else "")
    if cache_key not in _PLACEMENTS:
        text = " ".join(c["text"] for c in corpus.chunk_range(key, 0, min(n, 4000)))
        above = below = 0
        for pos, _ in captions(text):
            rows_before = has_table_rows(text[max(0, pos - 300):pos])
            rows_after = has_table_rows(text[pos:pos + 900])
            above += rows_after and not rows_before
            below += rows_before and not rows_after
        _PLACEMENTS[cache_key] = "below" if below > above else "above"
    return _PLACEMENTS[cache_key]


def label_tables(corpus, chunks: list[dict]) -> list[dict]:
    """Name the table a passage's leading rows belong to, and start every table
    caption inside a passage on a new paragraph."""
    out = []
    for c in chunks:
        key, cid, text = c.get("key"), c.get("chunk_id"), c["text"]
        caps = captions(text)
        lead_end = caps[0][0] if caps else len(text)
        # Rows before the first caption in the passage belong to a table that started earlier.
        if key is not None and cid is not None and has_table_rows(text[:lead_end]):
            if caption_placement(corpus, key) == "above":
                before = " ".join(b["text"] for b in
                                  corpus.chunk_range(key, max(0, cid - TABLE_SCAN), cid))
                if c.get("lead"):               # the lead-in is part of the passage, not before it
                    at = before.rfind(c["lead"].lstrip("…")[:60])
                    before = before[:at] if at >= 0 else before
                found = captions(before)
                owner = found[-1][1] if found else ""
            else:
                after = corpus.chunk_range(key, cid + 1, cid + 1 + TABLE_SCAN)
                found = caps or captions(" ".join(a["text"] for a in after))
                owner = found[0][1] if found else ""
            if owner:
                c = dict(c, table=owner)
                text = f"[The table rows in this passage before any new caption are from {owner}]\n" + text
                caps = [(p + len(text) - len(c["text"]), t) for p, t in caps]
        for pos, _ in reversed(caps):
            if pos > 0:
                text = text[:pos].rstrip() + "\n\n" + text[pos:]
        out.append(dict(c, text=text))
    return out


# Questions about who wrote a document: names and affiliations are on the title
# page, which a similarity search rarely finds ("who are the authors?" does not
# resemble a list of names and addresses).
_ASKS_ABOUT_AUTHORS = [
    re.compile(r"\bwho\b[^?]*\b(authors?|wrote|written|writers?|authored|researchers)\b", re.I),
    re.compile(r"\b(authors?|writers?|researchers)\b[^?]*\b(names?|based|located|affiliat\w*|"
               r"universit\w*|institutions?|countr\w*|organi[sz]ations?)\b", re.I),
    re.compile(r"\b(affiliations?|written by|authored by)\b", re.I),
    re.compile(r"\bwho (wrote|authored|is behind|published)\b", re.I),
]
_NOT_A_NAME = {"the", "and", "who", "are", "where", "what", "which", "they", "their", "based",
               "wrote", "written", "author", "authors", "paper", "papers", "document", "documents",
               "file", "pdf", "article", "study", "report", "book", "from", "this", "that",
               "with", "about", "affiliation", "affiliations", "located", "names", "name", "were",
               "was", "for", "how", "does", "did", "list", "give", "tell"}
TITLE_PAGE = 3          # chunks taken from the start of a document for author questions


def asks_about_authors(question: str) -> bool:
    return any(p.search(question) for p in _ASKS_ABOUT_AUTHORS)


def named_documents(question: str, keys: list[str], results: list[dict]) -> list[str]:
    """The documents a question refers to ("the SDN paper"), by words in their
    file names; otherwise those the search found, or all when there are few."""
    words = {w for w in re.findall(r"[a-z0-9]+", question.lower())
             if len(w) >= 3 and w not in _NOT_A_NAME}
    scores = {k: len(words & set(re.findall(r"[a-z0-9]+", os.path.basename(k).lower())))
              for k in keys}
    best = max(scores.values(), default=0)
    if best > 0:
        return [k for k in keys if scores[k] == best]
    if len(keys) <= 3:
        return list(keys)
    return list(dict.fromkeys(r["key"] for r in results if r.get("key") in scores))[:2]


# ── questions that name documents ("the quiet meadows paper") ───────────────
# A document is named by words from its title or file name placed before a
# word for a document: "the SDN paper", "the Trust-Weighted framework", "the CCDE
# guide", or after one: "the paper on quiet meadows".
_DOC_NOUN = r"(papers?|books?|guides?|articles?|stud(?:y|ies)|documents?|reports?|surveys?|frameworks?|systems?|pdfs?)"
# A word of a name: not a function or question word ("the authors of the SDN paper"
# names "SDN", not "authors of the SDN").
_NAME_WORD = r"(?:(?!(?:the|a|an|of|in|on|from|by|for|to|or|with|what|which|how|where|when|who|why|is|are|was|were|did|does|do|use|used|say|says)\b)[\w.+-]+)"
_NAMED_BEFORE = re.compile(rf"\b(?:the|in|from|of|by)\s+((?:{_NAME_WORD}\s+){{1,6}}?){_DOC_NOUN}\b", re.I)
_NAMED_AFTER = re.compile(rf"\b{_DOC_NOUN}\s+(?:on|about|called|titled|named)\s+((?:{_NAME_WORD}\s*){{1,5}})", re.I)
# Words in such a phrase that describe rather than name: "the whole SPCOR book".
_NOT_NAMING = {"whole", "entire", "proposed", "same", "other", "first", "second", "third", "last",
               "new", "original", "above", "previous", "earlier", "both", "two", "three", "each",
               "every", "your", "my", "our", "its", "their", "this", "that", "these", "those",
               "current", "indexed", "uploaded", "given", "following", "main", "overall", "full",
               "cited", "related"}
MAX_NAMED = 4           # a phrase matching more documents than this doesn't name one


def _name_words(text: str) -> set[str]:
    from .keyword import stem
    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())}


def _phrases(question: str) -> list[tuple[set[str], bool]]:
    """(name words, plural) for each name in the question. "the SDN and
    hierarchical papers" gives two names, each naming one document."""
    from .keyword import stem
    out = []
    for pattern, words_at, noun_at in ((_NAMED_BEFORE, 1, 2), (_NAMED_AFTER, 2, 1)):
        for m in pattern.finditer(question):
            noun = m.group(noun_at).lower()
            plural = noun == "studies" or (noun.endswith("s") and noun != "pdfs")
            parts = re.split(r"\s+and\s+|,\s*", m.group(words_at).strip(), flags=re.I)
            for part in parts:
                words = {stem(w) for w in re.findall(r"[a-z0-9]+", part.lower()) if w not in _NOT_NAMING}
                if words:
                    out.append((words, plural and len(parts) == 1))
    return out


def documents_named(question: str, titles: dict[str, str]) -> list[str]:
    """The documents a question names, in the order it names them. `titles` is
    {document key: title}; a document's name is its title plus its file name.
    A phrase picks the documents whose names contain all its words: one
    document, or several for a plural ("the RAG papers")."""
    if len(titles) < 2:
        return []
    names = {k: _name_words(f"{t} {os.path.splitext(os.path.basename(k))[0]}") for k, t in titles.items()}
    found = []
    for phrase, plural in _phrases(question):
        matching = [k for k, words in names.items() if phrase <= words]
        # "the RAG papers" may name several; "the cert guide" in a shelf of cert guides names none.
        if not matching or len(matching) > (MAX_NAMED if plural else 1):
            continue
        found += [k for k in matching if k not in found]
    return found


def title_page_chunks(corpus, key) -> list[dict]:
    """The first chunks of a document, up to its abstract."""
    head = corpus.chunk_range(key, 0, min(TITLE_PAGE, corpus.chunk_count(key)))
    out = []
    for i, ch in enumerate(head):
        if i > 0 and _ABSTRACT.search(ch["text"]) and not is_front_matter(ch["text"]):
            break
        out.append(dict(ch, key=key, section="title page"))
    return out


# ── depth and breadth ─────────────────────────────────────────────────────────
# "Explain how X works" needs one section read further; "Which chapters cover X?"
# needs short passages from many sections rather than several from one.
_DEPTH = re.compile(
    r"\b(explain\w*|elaborate|in (?:more |greater |full )?detail|step[- ]by[- ]step|walk (?:me )?through|"
    r"describe\w* how|how (?:does|do|is|are|did|was|were)\b[^?]*\b(?:work|works|done|performed|"
    r"implemented|calculated|computed|combined|built|configured|structured|organi[sz]ed))\b", re.I)
_BREADTH = re.compile(
    r"\b(?:which|what|in which) (?:chapters?|sections?|parts?)\b"
    r"|\bwhere\b[^?]*\b(?:discuss\w*|cover\w*|mention\w*|describ\w*|explain\w*|talk\w*|introduc\w*|"
    r"address\w*|appear\w*)\b"
    r"|\b(?:across|throughout)\b|\ball the (?:places|sections|chapters)\b", re.I)
DEPTH_EXTRA = 2         # further chunks of the same section added to the best passage
PER_SECTION = 2         # passages per section for breadth questions
BREADTH_EXTRA = 2       # breadth questions get this many passages more than top_k
FOLLOWUP_EXTRA = 2      # passages found with the previous answer, for follow-ups


def asks_for_depth(question: str) -> bool:
    return bool(_DEPTH.search(question)) and not _BREADTH.search(question)


def asks_for_breadth(question: str) -> bool:
    return bool(_BREADTH.search(question))


def spread(chunks: list[dict], n: int, per_section: int = PER_SECTION) -> list[dict]:
    """The best `n` passages with at most `per_section` from any one section
    (of any one document); the rest fill up only if there are too few sections."""
    from .structure import section_of
    taken, counts, rest = [], {}, []
    for ch in chunks:
        where = (ch.get("key") or ch["source"], section_of(ch.get("heading", "")) or ch.get("page"))
        if counts.get(where, 0) < per_section:
            counts[where] = counts.get(where, 0) + 1
            taken.append(ch)
        else:
            rest.append(ch)
    return (taken + rest)[:n]


def with_more_of_section(corpus, chunks: list[dict], extra: int = DEPTH_EXTRA) -> list[dict]:
    """Extend the best passage with the chunks after it, as long as they stay in
    its section (with_following_text has already added the first one)."""
    if not chunks:
        return chunks
    first = chunks[0]
    key, cid = first.get("key"), first.get("chunk_id")
    if key is None or cid is None:
        return chunks
    texts = {c["text"] for c in chunks}
    layout = doc_layout(corpus, key)
    text = first["text"]
    for nxt in corpus.chunk_range(key, cid + 2, cid + 2 + extra):
        if (nxt.get("heading", "") != first.get("heading", "") or nxt["text"] in texts
                or is_boilerplate(nxt["text"]) or in_back_matter(layout, nxt.get("chunk_id"), include_appendix=False)):
            break
        text = merge_overlapping(text, nxt["text"])
    return [dict(first, text=text)] + chunks[1:] if text != first["text"] else chunks


def corpus_titles(corpus) -> dict:
    """{document key: title}; file names stand in when the corpus has no titles."""
    if corpus is None:
        return {}
    if hasattr(corpus, "titles"):
        return corpus.titles()
    return {k: os.path.splitext(os.path.basename(k))[0] for k in corpus.keys()}


def _named_note(keys: list[str], titles: dict) -> str:
    names = "; ".join(f"{os.path.basename(k)} (\"{titles[k]}\")" if titles.get(k) else os.path.basename(k)
                      for k in keys)
    if len(keys) == 1:
        return (f"The question names one document: {names}. The passages come from it only. "
                "Answer in this document's own terms: if the question describes something with a "
                "term this document does not use (it may come from another document), say so, "
                "then give what this document does say.")
    return (f"The question names {len(keys)} documents: {names}. Passages from each are included. "
            "Keep apart what each document says, and say which one each point comes from.")


def _merge_best(*lists: list[dict]) -> list[dict]:
    merged = {}
    for r in (r for rows in lists for r in rows):
        if r["text"] not in merged or r["score"] > merged[r["text"]]["score"]:
            merged[r["text"]] = r
    return list(merged.values())


def prepare(client: OllamaClient, settings: Settings, model: str, question: str,
            history: list[dict], search: Callable[[list[float], int], list[dict]],
            corpus=None) -> Prepared:
    """`corpus`, when given, enables broad questions, named documents and the
    reading of neighbouring chunks. It needs: keys() -> document keys;
    search(vector, n, key) -> results within one document; chunk_count(key) ->
    number of chunks; chunk_range(key, start, stop) -> chunks in document order
    (with chunk_id). Optional: keyword_search(query, n[, key]) and titles()."""
    num_ctx = context_window(client, settings, model)
    history = [h for h in (history or []) if h.get("q") and h.get("a")]
    keys = corpus.keys() if corpus is not None else []
    # The conversation is used only for questions that depend on it. A complete
    # question is answered from the documents alone, so claims from earlier
    # answers can't leak in as if they were facts.
    if not looks_like_followup(question, one_document=len(keys) == 1):
        history = []
    search_query = question
    if history and settings.rewrite_followups and not is_broad(question):
        search_query = rewrite_followup(client, model, question, history, num_ctx)
    vector = embed_query(client, settings, search_query)

    titles = corpus_titles(corpus) if len(keys) >= 2 else {}
    named = [k for k in documents_named(question, titles) or documents_named(search_query, titles) if k in keys]

    if len(keys) >= 2 and (is_broad(question) or is_broad(search_query)):
        scope = named or keys
        chunks = broad_chunks(corpus, vector, settings, scope)
        names = list(dict.fromkeys(ch["source"] for ch in chunks))
        if len(names) == 1:
            note = (f"The passages come from one document: {names[0]}. Passages marked 'abstract' or "
                    "'conclusion' state its own findings.")
        else:
            note = (f"The passages come from {len(names)} documents: " + "; ".join(names) +
                    ". Passages marked 'abstract' or 'conclusion' state each document's own findings. "
                    "Cover each document separately, then compare them if asked.")
        text, used, turns = P.build_prompt(question, chunks, history, num_ctx,
                                           settings.answer_reserve, settings.history_turns, note=note)
        return Prepared(text, used, search_query, num_ctx, turns, broad=len(names) > 1)

    hybrid = settings.keyword_search and corpus is not None and hasattr(corpus, "keyword_search")
    breadth = asks_for_breadth(question)
    top_k = settings.top_k + (BREADTH_EXTRA if breadth else 0)
    scopes = named or [None]
    each = top_k if len(scopes) == 1 else max(3, math.ceil(top_k / len(scopes)))
    pool = each * (3 if breadth else 2 if hybrid else 1)
    vectors = [vector] + ([embed_query(client, settings, question)] if search_query != question else [])
    # A follow-up ("explain the second step") is about what the last answer said:
    # passages close to that answer are searched for as well.
    answer_vector = None
    if history:
        answer_vector = embed_query(client, settings, f"{history[-1]['a'][:1500]}\n{question}")
    keyword_query = question if search_query == question else f"{search_query} {question}"

    def similar(vec, n, key):
        return search(vec, n) if key is None else corpus.search(vec, n, key)

    def keyword(n, key):
        if not hybrid:
            return []
        if key is None:
            return corpus.keyword_search(keyword_query, n)
        try:
            return corpus.keyword_search(keyword_query, n, key)
        except TypeError:                     # a corpus whose keyword search has no key
            return [r for r in corpus.keyword_search(keyword_query, n * 3) if r.get("key") == key][:n]

    def outside_reference_lists(rows):
        # Reference lists match many questions on wording alone ("FWOS" came from one).
        if corpus is None or _ASKS_ABOUT_REFERENCES.search(question):
            return rows
        return [r for r in rows if r.get("key") is None or not in_back_matter(
            doc_layout(corpus, r["key"]), r.get("chunk_id"), include_appendix=False)]

    groups, first_results = [], []
    for key in scopes:
        # A rewrite can drift from what was asked; search with the original
        # wording too and keep the best of both.
        results = outside_reference_lists(_merge_best(*(similar(v, pool, key) for v in vectors)))
        words = outside_reference_lists(keyword(pool, key))
        first_results += results
        want = each * 2 if breadth else each
        found = fuse(results, words, want, settings.min_score) if hybrid else \
            select(results, want, settings.min_score)
        if breadth:
            found = spread(found, each)
        if answer_vector is not None:
            seen = {r["text"] for r in found}
            extra = [r for r in outside_reference_lists(similar(answer_vector, each, key))
                     if r["text"] not in seen and r["score"] >= settings.min_score]
            found += extra[:FOLLOWUP_EXTRA]
        groups.append(found)
    chunks = []
    for i in range(max((len(g) for g in groups), default=0)):
        chunks += [g[i] for g in groups if i < len(g)]

    if corpus is not None and (asks_about_authors(question) or asks_about_authors(search_query)):
        who = named or named_documents(question + " " + search_query, corpus.keys(), first_results)
        front = [ch for key in who for ch in title_page_chunks(corpus, key)]
        seen = {ch["text"] for ch in front}
        chunks = front + [ch for ch in chunks if ch["text"] not in seen]
    if corpus is not None:
        chunks = with_preceding_text(corpus, with_following_text(corpus, chunks))
        if asks_for_depth(question) and len(scopes) == 1:
            chunks = with_more_of_section(corpus, chunks)
        chunks = label_tables(corpus, chunks)
    if settings.grade_chunks and chunks:
        chunks = grade_chunks(client, model, search_query, chunks, num_ctx)
    notes = []
    if named:
        notes.append(_named_note(named, titles))
    if breadth:
        notes.append("The question asks where a topic is covered. The passages come from different "
                     "sections, named in their labels; say which sections cover it.")
    text, used, turns = P.build_prompt(question, chunks, history, num_ctx,
                                       settings.answer_reserve, settings.history_turns,
                                       note="\n".join(notes))
    return Prepared(text, used, search_query, num_ctx, turns)


def generation_options(num_ctx: int) -> dict:
    return {"temperature": 0.1, "num_ctx": num_ctx}
