"""Document structure: the title of a document and the section each chunk is in.

A passage is easier to use when it is known to come from "6. Conclusion,
limitations and future work" rather than from "page 8", both for the model
(which section is this?) and for the search (a passage's heading is part of
what it is about). Sections come from the PDF's outline (bookmarks) when it
has one, otherwise from numbered headings found in the text. Titles come from
the PDF's metadata, otherwise from its first page.
"""
import os
import re

from .chunking import Chunk, joined_pages

MAX_LEVELS = 3          # heading path depth kept: "Part II › Chapter 5 › Section"
MAX_PART = 70           # characters per heading in the path
SEP = " › "

_NAMED = re.compile(r"^(Abstract|Introduction|Background|Related [Ww]ork|Methods?|Methodology|Results|"
                    r"Discussion|Conclusions?|Limitations|Future [Ww]ork|Acknowledg(?:e)?ments?|"
                    r"References|Bibliography|Appendix(?: [A-Z])?|Glossary|Index|ABSTRACT|INTRODUCTION|"
                    r"CONCLUSIONS?|ACKNOWLEDG(?:E)?MENTS?|REFERENCES|APPENDIX(?: [A-Z])?)$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(\S.{1,100})$")
_ROMAN = re.compile(r"^([IVX]{1,5})\.\s+([A-Z][A-Z0-9 ,:&'()\-–]{2,90})$")
_LETTERED = re.compile(r"^([A-H])\.\s+([A-Z]\S*(?:\s+\S+){1,11})$")
_MARKDOWN = re.compile(r"^(#{1,4})\s+(\S.{1,100})$")
_APPENDIX = re.compile(r"^(Appendix|APPENDIX)\s+([A-Z])\s*[.:]\s*([A-Z].{1,80})$")
_ROMANS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


def _clean(title: str) -> str:
    return " ".join(title.split()).strip(" .:")


_TRAILING = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or",
             "the", "to", "with", "when", "making", "each", "that", "which", "between", "based"}


def _looks_like_title(text: str) -> bool:
    """A heading, not a sentence, list item, wrapped line or running header."""
    words = text.split()
    return (1 <= len(words) <= 14 and text[0].isupper() and not text.rstrip().endswith((".", ",", ";", "-", ":"))
            and words[-1].lower() not in _TRAILING
            and "et al" not in text and not re.search(r"\d{4,}|https?://|@", text))


# ── headings from the PDF outline ─────────────────────────────────────────────

def outline_headings(path: str, title: str = "") -> list[tuple[int, int, str]]:
    """(page, level, heading) from a PDF's bookmarks, in document order."""
    if not path.lower().endswith(".pdf"):
        return []
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        out = []

        def walk(items, level):
            for item in items:
                if isinstance(item, list):
                    walk(item, level + 1)
                    continue
                try:
                    page = reader.get_destination_page_number(item) + 1
                except Exception:
                    continue
                out.append((page, level, _clean(item.title or "")))

        walk(reader.outline, 1)
    except Exception:
        return []
    # A top entry that is just the document's own title isn't a section.
    norm = lambda t: re.sub(r"\W+", " ", t).lower().strip()
    if title and out and norm(out[0][2])[:40] == norm(title)[:40]:
        top = out[0][1]
        out = [(p, lvl - 1 if lvl > top else lvl, t) for p, lvl, t in out[1:]]
    return [(p, max(1, lvl), t) for p, lvl, t in out if t]


# ── headings found in the text ────────────────────────────────────────────────

def text_headings(pages) -> list[tuple[int | None, int, str]]:
    """(page, level, heading) for numbered and well-known headings on their own
    line. Numbered headings must continue the numbering (4 after 3, 3.2 after
    3.1), which keeps out numbered lists, footnotes and table rows."""
    lines_per_page = [(page, [l.strip() for l in (text or "").splitlines() if l.strip()]) for page, text in pages]
    seen: dict[str, int] = {}
    for _, lines in lines_per_page:
        for line in set(lines):
            seen[line] = seen.get(line, 0) + 1
    out = []
    numbers: list[int] = []            # current position in arabic numbering: [3, 2] = "3.2"
    roman, letter, in_appendix = 0, 0, False
    for page, lines in lines_per_page:
        for line in lines:
            if seen[line] >= 3 or not _looks_like_title(re.sub(r"^\S+\s+", "", line) or line):
                if not (_NAMED.match(line) or _APPENDIX.match(line)):
                    continue
            if _NAMED.match(line):
                out.append((page, 1, line.strip()))
                continue
            m = _APPENDIX.match(line)
            if m and _looks_like_title(m.group(3)):
                in_appendix = True               # numbered lines after this are lists, not sections
                out.append((page, 1, f"Appendix {m.group(2)}. {_clean(m.group(3))}"))
                continue
            m = _MARKDOWN.match(line)
            if m:
                out.append((page, len(m.group(1)), _clean(m.group(2))))
                continue
            m = _NUMBERED.match(line)
            if m and not in_appendix and _looks_like_title(m.group(2)):
                parts = [int(x) for x in m.group(1).split(".")]
                expected = parts[:-1] == numbers[:len(parts) - 1] and (
                    parts[-1] == (numbers[len(parts) - 1] + 1 if len(numbers) >= len(parts) else 1))
                if expected or (not numbers and parts == [1]):
                    numbers = parts
                    out.append((page, len(parts), f"{m.group(1)}. {_clean(m.group(2))}"))
                continue
            m = _ROMAN.match(line)
            if m and roman < len(_ROMANS) and m.group(1) == _ROMANS[roman]:
                roman, letter = roman + 1, 0
                out.append((page, 1, f"{m.group(1)}. {_clean(m.group(2)).title()}"))
                continue
            m = _LETTERED.match(line)
            if m and roman and ord(m.group(1)) - ord("A") == letter and _looks_like_title(m.group(2)):
                letter += 1
                out.append((page, 2, f"{m.group(1)}. {_clean(m.group(2))}"))
    return out


# ── assigning headings to chunks ──────────────────────────────────────────────

def _positions(headings, pages) -> list[tuple[int, int, str]]:
    """Each heading's offset in the joined text (as used by Chunk.start)."""
    text, starts, numbers = joined_pages(pages)
    lower = text.lower()
    page_start = {n: s for n, s in zip(numbers, starts)}
    page_end = {n: e for n, e in zip(numbers, starts[1:] + [len(text)])}
    out, cursor = [], 0
    for page, level, title in headings:
        begin = max(cursor, page_start.get(page, cursor))
        probe = " ".join(title.split()).lower()
        probe = re.sub(r"^[\dIVXA-H.]+\s+", "", probe)[:40] or probe
        at = lower.find(probe, begin)
        if at < 0 or at > page_end.get(page, len(text)) + 200:
            at = begin
        out.append((at, level, title))
        cursor = at
    return out


def assign_headings(chunks: list[Chunk], pages, headings) -> None:
    """Set chunk.heading to the section path at the middle of each chunk."""
    marks = _positions(headings, pages)
    i, stack = 0, []
    for ch in chunks:
        middle = ch.start + len(ch.text) // 2
        while i < len(marks) and marks[i][0] <= middle:
            _, level, title = marks[i]
            stack = stack[:level - 1] + [title[:MAX_PART].rstrip()]
            i += 1
        ch.heading = SEP.join(stack[-MAX_LEVELS:])


# ── titles ────────────────────────────────────────────────────────────────────

_BAD_TITLE = re.compile(r"^(untitled|microsoft word|document\d*|title|slide \d+)\b|\.(docx?|pdf|tex)$", re.I)


def document_title(path: str, pages) -> str:
    """The document's title: PDF metadata if sensible, else the first lines of
    its first page that look like a title, else the file name."""
    if path.lower().endswith(".pdf"):
        try:
            import pypdf
            meta = pypdf.PdfReader(path).metadata or {}
            title = _clean(str(meta.get("/Title") or ""))
            if len(title.split()) >= 3 and not _BAD_TITLE.search(title):
                return title
        except Exception:
            pass
    first = next((t for _, t in pages if t and t.strip()), "")
    lines = [l.strip() for l in first.splitlines() if l.strip()][:25]
    title_lines = []
    for line in lines:
        words = line.split()
        wordy = len(re.findall(r"[A-Za-z]{2,}", line)) >= max(1, len(words) - 1)
        if not title_lines:
            if (len(words) >= 3 and wordy and line[0].isupper() and not re.search(
                    r"https?://|www\.|journal|contents|available|homepage|article|©|copyright", line, re.I)):
                title_lines.append(line)
        elif (len(words) <= 12 and wordy and "," not in line and len(" ".join(title_lines)) < 160
              and not any(p.match(line) for p in (_NAMED, _NUMBERED, _ROMAN, _MARKDOWN, _APPENDIX))):
            title_lines.append(line)
        else:
            break
    title = re.sub(r"\s+", " ", " ".join(title_lines)).strip()
    title = re.sub(r"(?<=[a-z])[A-Z∗*☆★]$", "", title)        # footnote mark glued to the last word
    return title[:200] or os.path.splitext(os.path.basename(path))[0]


def analyse(path: str, pages, chunks: list[Chunk]) -> str:
    """Give every chunk its heading; return the document's title. Structure is
    a help, not a requirement: if anything goes wrong the chunks keep no heading."""
    try:
        title = document_title(path, pages)
    except Exception:
        title = os.path.splitext(os.path.basename(path))[0]
    try:
        headings = outline_headings(path, title) or text_headings(pages)
        if headings:
            assign_headings(chunks, pages, headings)
    except Exception:
        for ch in chunks:
            ch.heading = ""
    return title


def load(path: str, chunk_size: int, overlap: int, report: dict | None = None):
    """Read, chunk and analyse a document: (pages, chunks with headings, title)."""
    from .chunking import chunk_pages
    from .extract import extract_pages
    pages = extract_pages(path, report)
    chunks = chunk_pages(pages, chunk_size, overlap)
    return pages, chunks, (analyse(path, pages, chunks) if chunks else "")


def deepest(heading: str) -> str:
    """The last part of a heading path: "4. Method › 4.2 Data" → "4.2 Data"."""
    return heading.rsplit(SEP, 1)[-1] if heading else ""


def section_of(heading: str) -> str:
    """The part of a heading path that counts as "the section" for depth and
    breadth questions: the top two levels."""
    return SEP.join(heading.split(SEP)[:2]) if heading else ""
