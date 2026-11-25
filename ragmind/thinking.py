"""Hide a reasoning model's "thinking" so only its answer is shown.

Ollama separates the reasoning of models it knows (gpt-oss, qwen3, deepseek-r1
from its library) into a field RagMind ignores. Models imported from Hugging
Face are not separated: their reasoning arrives inside the answer, either as
<think>…</think> or (when the chat template already opened the tag) as plain
text followed by a lone </think>.
"""

import re
from typing import Iterable, Iterator

from .ollama import ModelInfo

OPEN, CLOSE = "<think>", "</think>"
_REASONING_NAME = re.compile(r"(reason|think|deepseek-r1|[-_:/]r1\b|qwq|magistral)", re.I)


def reasons_unseparated(info: ModelInfo) -> bool:
    """A reasoning model whose thinking Ollama does NOT separate for us."""
    if info.capabilities and "thinking" in info.capabilities:
        return False
    return bool(_REASONING_NAME.search(info.name))


class ThinkingFilter:
    """Feed answer tokens in; get back only the text meant for the reader.

    hold=True (for reasoning models): show nothing until </think> appears; if
    it never does, the whole text is shown at the end. hold=False: text is
    passed through as it streams, except a leading <think>…</think> block.
    """

    def __init__(self, hold: bool = False):
        self.buf = ""
        self.state = "hold" if hold else "start"

    @property
    def thinking(self) -> bool:
        return self.state in ("hold", "inside")

    def feed(self, tok: str) -> str:
        if self.state == "pass":
            return tok
        if self.state == "after":               # drop blank lines after </think>
            tok = tok.lstrip()
            if tok:
                self.state = "pass"
            return tok
        self.buf += tok
        if self.state == "start":
            head = self.buf.lstrip()
            if head.startswith(OPEN):
                self.state = "inside"
            elif OPEN.startswith(head):        # could still become "<think>"
                return ""
            else:
                self.state, out, self.buf = "pass", self.buf, ""
                return out
        # "inside" or "hold": wait for the closing tag
        end = self.buf.find(CLOSE)
        if end == -1:
            return ""
        out, self.buf = self.buf[end + len(CLOSE):].lstrip(), ""
        self.state = "pass" if out else "after"
        return out

    def finish(self) -> str:
        """Whatever is left when the answer ends."""
        out, self.buf = self.buf, ""
        if self.state in ("pass", "after"):
            return out
        self.state = "pass"
        # Never closed: better to show the text (minus the tag) than nothing.
        return out.replace(OPEN, "", 1).strip()


def visible(tokens: Iterable[str], hold: bool = False) -> Iterator[str]:
    f = ThinkingFilter(hold)
    for tok in tokens:
        out = f.feed(tok)
        if out:
            yield out
    rest = f.finish()
    if rest:
        yield rest


def strip(text: str) -> str:
    """Non-streaming version (for short helper replies)."""
    if CLOSE in text:
        return text.split(CLOSE, 1)[1].strip()
    return text.replace(OPEN, "").strip()
