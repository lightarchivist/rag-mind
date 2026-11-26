"""Prompt building with a token budget, so nothing important gets cut off.

Ollama silently drops the *start* of a prompt that is longer than the context
window. So the budget is enforced here: the rules and the question are always
kept, then passages (best first), then as much recent conversation as fits.
"""

SYSTEM = """You are a document analysis assistant.
Answer questions ONLY using the retrieved passages below.
Rules:
1. Base your answer solely on the provided passages. Do not use outside knowledge.
2. If the answer is not in the passages, say: "This information is not found in the indexed documents."
3. Cite the passages you use by their number in square brackets, e.g. [2].
4. When asked which or what items (datasets, models, methods, authors...), name every one the passages mention.
5. When quoting numbers from a table, say which row and column (e.g. which model and dataset) they belong to, exactly as in the passage.
6. Check the question's assumptions. If it attributes something to a document whose passages do not say it (for example, the passages come from a different document), say so instead of answering as if it were true.
7. Keep the authors' own work apart from work they cite: "Ref. [89] proposes X" means other researchers proposed X.
8. The conversation so far is context for follow-up questions, not a source of facts.
9. When asked how many there are and the passages give no total, list the ones the passages name and count them, saying that this is how many the passages mention."""

NO_PASSAGES = "(No passages matched this question.)"
MAX_ANSWER_CHARS = 1500        # earlier answers are shortened to this in the prompt


def estimate_tokens(text: str) -> int:
    # About 4 characters per token for English; 3 is a safe over-estimate that
    # also covers code, numbers and other languages.
    return len(text) // 3 + 1


def cite(chunk: dict) -> str:
    page = chunk.get("page")
    return f"{chunk['source']}, p. {page}" if page else chunk["source"]


# Some models (gpt-oss) cite as 【2】; shown as [2], like the source tags.
_CITATION_BRACKETS = str.maketrans({"【": "[", "】": "]"})


def tidy_citations(text: str) -> str:
    return text.translate(_CITATION_BRACKETS)


SEP = " › "     # between heading levels (as in structure.SEP)


def place(chunk: dict) -> str:
    """Where in the document a passage is, briefly: its own heading and, for
    passages picked as a document's abstract or conclusion, that role
    ("4.2 Training data", "Abstract, abstract" is shortened to "Abstract")."""
    heading = (chunk.get("heading") or "").rsplit(SEP, 1)[-1]
    role = chunk.get("section", "")
    if role and heading.lower().endswith(role.lower()):
        role = ""
    return ", ".join(p for p in (heading, role) if p)


def group_sources(chunks: list[dict]) -> list[tuple[list[int], str, str]]:
    """Passages from the same document, page and section share one entry:
    ([1, 3, 5], "paper.pdf, p. 1", "abstract"), in order of first appearance."""
    groups: dict = {}
    for i, ch in enumerate(chunks, 1):
        groups.setdefault((cite(ch), place(ch)), []).append(i)
    return [(numbers, c, where) for (c, where), numbers in groups.items()]


def format_passage(i: int, chunk: dict) -> str:
    # No similarity score here: models copied it into answers ("relevance: 0.78").
    heading = f", in {chunk['heading']}" if chunk.get("heading") else ""
    section = f", {chunk['section']}" if chunk.get("section") else ""
    return f"[{i}] (source: {cite(chunk)}{heading}{section})\n{chunk['text']}"


def documents_note(chunks: list[dict]) -> str:
    """The titles of the documents the passages come from, so the model knows
    that "the quiet meadows paper" is brightwater-2031.pdf."""
    titles = {}
    for ch in chunks:
        title = (ch.get("title") or "").strip()
        if title and ch["source"] not in titles and title.lower() != ch["source"].rsplit(".", 1)[0].lower():
            titles[ch["source"]] = title
    if not titles:
        return ""
    return "Document titles: " + "; ".join(f"{src} = \"{t}\"" for src, t in titles.items()) + "."


def _turn(h: dict) -> str:
    answer = h.get("a", "")
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[:MAX_ANSWER_CHARS] + " […]"
    return f"User: {h.get('q', '')}\nAssistant: {answer}"


def build_prompt(question: str, chunks: list[dict], history: list[dict],
                 ctx_tokens: int, answer_reserve: int = 1024, history_turns: int = 4,
                 note: str = ""):
    """Return (prompt, chunks_used, turns_used) fitting in ctx_tokens - answer_reserve.
    `note` is an extra line placed just above the passages."""
    budget = max(ctx_tokens - answer_reserve, 256)
    tail = f"User: {question}\nAssistant:"
    used = (estimate_tokens(SYSTEM) + estimate_tokens(tail) + estimate_tokens(note)
            + estimate_tokens(documents_note(chunks)) + 40)

    recent = [h for h in (history or [])[-history_turns:] if h.get("q")]
    turns = [_turn(h) for h in recent]
    # Keep room for the latest turn, which follow-up questions depend on.
    last_cost = estimate_tokens(turns[-1]) if turns else 0

    passages = []
    for ch in chunks:
        cost = estimate_tokens(format_passage(len(passages) + 1, ch))
        if used + cost + last_cost > budget:
            break
        passages.append(ch)
        used += cost

    kept_turns = []
    for t in reversed(turns):               # newest first
        cost = estimate_tokens(t)
        if used + cost > budget:
            break
        kept_turns.insert(0, t)
        used += cost

    titles = documents_note(passages)
    note = "\n".join(p for p in (titles, note) if p)
    context = "\n\n".join(format_passage(i, ch) for i, ch in enumerate(passages, 1)) or NO_PASSAGES
    parts = [SYSTEM]
    if kept_turns:
        parts.append("=== CONVERSATION SO FAR ===\n" + "\n".join(kept_turns))
    lead = f"{note}\n\n" if note else ""
    parts.append(f"=== RETRIEVED PASSAGES ===\n{lead}{context}\n=== END PASSAGES ===")
    parts.append(tail)
    return "\n\n".join(parts), passages, len(kept_turns)


# ── helper prompts (follow-up rewriting, passage grading) ─────────────────────

REWRITE = """Rewrite the user's latest question so it can be understood without the conversation.
Replace words like "it", "that", "the second one" with what they refer to.
Keep descriptions such as "the smallest model" or "the best method" as they are: do not replace them with a name, and do not guess which one they mean.
Do not add facts, names or numbers that the latest question does not refer to.
If the question is already self-contained, return it unchanged.
Reply with the rewritten question only, on one line.

Conversation:
{conversation}

Latest question: {question}
Rewritten question:"""


def build_rewrite_prompt(question: str, history: list[dict], turns: int = 2) -> str:
    convo = "\n".join(_turn(h) for h in history[-turns:])
    return REWRITE.format(conversation=convo, question=question)


GRADE = """Does this passage contain information that helps answer the question?
Reply with one word: yes or no.

Question: {question}

Passage:
{passage}

Answer:"""


def build_grade_prompt(question: str, passage: str) -> str:
    return GRADE.format(question=question, passage=passage)
