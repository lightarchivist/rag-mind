"""Split text into overlapping chunks, keeping track of which page each starts on."""

import bisect
from dataclasses import dataclass

SENTENCE_ENDS = (". ", "? ", "! ")


@dataclass
class Chunk:
    text: str
    chunk_id: int
    page: int | None = None        # page where the chunk starts (PDFs only)
    start: int = 0                 # position in the joined document text (see joined_pages)
    heading: str = ""              # section the chunk belongs to (see structure.py)


def joined_pages(pages) -> tuple[str, list[int], list]:
    """Whitespace-normalised pages joined with single spaces, with the offset
    where each page starts and its page number. Chunk.start uses these offsets."""
    starts, numbers, parts, offset = [], [], [], 0
    for number, raw in pages:
        text = " ".join((raw or "").split())
        if not text:
            continue
        if parts:
            offset += 1                     # the joining space
        starts.append(offset)
        numbers.append(number)
        parts.append(text)
        offset += len(text)
    return " ".join(parts), starts, numbers


def chunk_pages(pages, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    """`pages` is a list of (page_number_or_None, text)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text, starts, numbers = joined_pages(pages)

    def page_at(pos):
        i = bisect.bisect_right(starts, pos) - 1
        return numbers[i] if i >= 0 else None

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):                 # prefer to end at a sentence boundary
            for sep in SENTENCE_ENDS:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            lead = len(text[start:end]) - len(text[start:end].lstrip())
            chunks.append(Chunk(piece, len(chunks), page_at(start + lead), start + lead))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)   # always move forward
    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    return chunk_pages([(None, text)], chunk_size, overlap)
