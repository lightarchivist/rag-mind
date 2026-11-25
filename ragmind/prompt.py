"""Prompt building with a token budget, so nothing important gets cut off.

Ollama silently drops the *start* of a prompt that is longer than the context
window. So the budget is enforced here: the rules and the question are always
kept, then passages (best first), then as much recent conversation as fits.
"""

SYSTEM = """You are a document analysis assistant.
Answer questions ONLY using the retrieved passages below.
Rules:
1. Base your answer solely on the provided passages.
2. If the answer is not in the passages, say: "This information is not found in the indexed documents."
3. Cite the source filename (and page, when given) for the facts you use.
4. Do not use outside knowledge.
5. The conversation so far is context for follow-up questions, not a source of facts."""

NO_PASSAGES = "(No passages matched this question.)"
MAX_ANSWER_CHARS = 1500        # earlier answers are shortened to this in the prompt


def estimate_tokens(text: str) -> int:
    # About 4 characters per token for English; 3 is a safe over-estimate that
    # also covers code, numbers and other languages.
    return len(text) // 3 + 1


def cite(chunk: dict) -> str:
    page = chunk.get("page")
    return f"{chunk['source']}, p. {page}" if page else chunk["source"]


def format_passage(i: int, chunk: dict) -> str:
    # No similarity score here: models copied it into answers ("relevance: 0.78").
    return f"[{i}] (source: {cite(chunk)})\n{chunk['text']}"


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
    used = estimate_tokens(SYSTEM) + estimate_tokens(tail) + estimate_tokens(note) + 40

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
