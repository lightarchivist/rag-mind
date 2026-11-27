"""Text extraction. Returns a list of (page_number, text) so answers can cite
pages; formats without pages return a single (None, text) entry."""

import collections
import os
import re

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".rst", ".html", ".htm", ".epub"}

Pages = list[tuple[int | None, str]]


class ExtractionError(Exception):
    """A file could not be read. The message is meant for the user."""


# Some PDF fonts map the glyph for "s" to a minus sign (U+2212), so "session"
# is extracted as "−e−−ion" and "is" as "i−". Such a document can't be searched
# for any word with an "s" in it. A minus sign touching a letter is turned back
# into "s", but only in documents where this is clearly happening: many such
# signs, and few real "s" left. Real minus signs ("5 − 3") are never touched.
_BROKEN_S = re.compile(r"(?<=[A-Za-z])\u2212|\u2212(?=[a-z])")
BROKEN_S_MIN = 50
# The buyer's watermark on eBooks sold by Pearson/InformIT, on every page:
# "From the Library of <name> ptg12345678". It is noise in every passage.
# Any whitespace may separate the words: some PDFs use no-break spaces or put
# the name on the next line, which the normalised passage text no longer shows.
_FROM_LIBRARY = r"From\s+the\s+Library\s+of\s+"
_WATERMARK = re.compile(_FROM_LIBRARY + r"(?s:.{1,80}?)\s*ptg\d{6,}|\bptg\d{6,}\b")


# Typographic ligatures: "efﬁciency" contains one character "ﬁ", so it would
# not match "efficiency" in a search.
_LIGATURES = str.maketrans({"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
                            "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"})
# Download stamps on every page of papers from IEEE Xplore, naming the
# institution: "Authorized licensed use limited to: … Restrictions apply."
_STAMP = re.compile(r"Authorized licensed use limited to:[^\n]{0,250}?Restrictions apply\.?")
# Letter spacing that splits a capital from the rest of its word ("T able 1",
# "T o ensure", "V ery"), common in some journals' PDFs. Joined only where it
# happens throughout a document, and only where the result is a word: "T cells",
# "V is the set of nodes" and "Y cable" are real and stay as they are.
_SPLIT_CAPITAL = re.compile(r"\b([TVWY]) ([a-z]+)\b")
SPLIT_CAPITAL_MIN = 10


def _split_word_test(whole: str):
    """A test for one _SPLIT_CAPITAL match: join "T able" when "table" is a word
    elsewhere in the document, or when "able" isn't ("T eachers")."""
    vocab = set(re.findall(r"[a-z]+", _SPLIT_CAPITAL.sub(" ", whole).lower()))
    return lambda m: (m.group(1) + m.group(2)).lower() in vocab or m.group(2) not in vocab
# Small capitals in numbered headings come out as "VIII. C ONCLUSION".
_SPLIT_HEADING = re.compile(r"(\b[IVX]+\.\s+)([A-Z]) ([A-Z]{2,})\b")


# The same watermark without the "ptg" number: "From the Library of <Name>".
# It is removed only when one name repeats on many pages, which ordinary text
# ("From the Library of Congress", once) never does.
# The name: up to four words on one line (any spaces between them, no-break
# spaces included). Letters of any language, and lowercase particles: "José
# van der Berg".
_NAME_WORDS = r"(?:[^\W\d_][\w'’.-]*)(?:[^\S\n]+[^\W\d_][\w'’.-]*){0,3}"
_LIBRARY_OF = re.compile(_FROM_LIBRARY + "(" + _NAME_WORDS + ")")
# Some PDFs wrap the name: "From the Library of Jane", then "Doe" alone on the
# next line. The words that start the line after a name that ends its line:
_NEXT_LINE = re.compile(r"[^\S\n]*\n\s*(" + _NAME_WORDS + ")")
WATERMARK_MIN = 10
WRAPPED_SHARE = 0.9     # how often the same words must follow to be the rest of the name


def _watermark_names(whole: str) -> list[str]:
    """Names that follow "From the Library of" on many pages, longest form first.
    Every common form is kept ("Jane Doe Chapter", "Jane Doe", "Jane"): removing
    them longest first takes the whole name off every page, whatever follows it
    there (a running header such as "Chapter 10" on some pages only).
    A wrapped name ("Jane" alone at the end of the line, "Doe" on the next) is
    kept as one form with the line break in it ("Jane\nDoe"). Words count as
    the rest of the name only when they start the next line nearly every time
    that first word ends a line, and appear almost nowhere else: a running
    header there varies from page to page, and body text repeats elsewhere."""
    counts = collections.Counter()
    line_ends = collections.Counter()
    wrapped = collections.Counter()
    for m in _LIBRARY_OF.finditer(whole):
        words = m.group(1).split()
        for k in range(1, len(words) + 1):
            counts[" ".join(words[:k])] += 1
        if len(words) > 1:
            continue                       # a name of two or more words is complete
        nxt = _NEXT_LINE.match(whole, m.end())
        if nxt or whole[m.end():].lstrip(" \t ").startswith("\n"):
            line_ends[m.group(1)] += 1
        if nxt:
            rest = nxt.group(1).split()
            for k in range(1, len(rest) + 1):
                wrapped[(m.group(1), " ".join(rest[:k]))] += 1
    names = [n for n, c in counts.items() if c >= WATERMARK_MIN]
    names += [f"{name}\n{rest}" for (name, rest), c in wrapped.items()
              if c >= WATERMARK_MIN and c >= WRAPPED_SHARE * line_ends[name]
              and _occurrences(rest, whole) <= c + max(2, c // 10)]
    return sorted(names, key=len, reverse=True)


def _occurrences(words: str, whole: str) -> int:
    """How often these words appear in the text. The rest of a wrapped name
    appears almost only in the watermark; a header word ("Chapter") elsewhere too."""
    return len(re.findall(r"(?<!\w)" + r"\s+".join(map(re.escape, words.split())) + r"(?!\w)", whole))


def _owner_pattern(name: str) -> re.Pattern:
    """The watermark with this name, words on one line apart from a wrap ("\n")."""
    lines = [r"[^\S\n]+".join(map(re.escape, part.split())) for part in name.split("\n")]
    return re.compile(_FROM_LIBRARY + r"[^\S\n]*\n\s*".join(lines) + r"(?:\s*ptg\d{6,})?")


def repair(pages: Pages, report: dict | None = None) -> Pages:
    """Undo common PDF text problems: the broken-"s" font, ligatures, split
    capitals and small-capital headings; remove eBook watermarks and download
    stamps. `report` receives the counts."""
    whole = "".join(text for _, text in pages)
    broken = len(_BROKEN_S.findall(whole)) + len(re.findall(r"(?<=[A-Za-z]\u2212)\u2212(?![a-z])", whole))
    fix_s = broken >= BROKEN_S_MIN and broken >= 0.5 * whole.count("s")
    joinable = _split_word_test(whole)
    join_capitals = sum(1 for m in _SPLIT_CAPITAL.finditer(whole) if joinable(m)) >= SPLIT_CAPITAL_MIN
    owners = [_owner_pattern(n) for n in _watermark_names(whole)]
    counts = {"watermarks": 0, "stamps": 0, "ligatures": 0, "split_words": 0}
    out = []
    for page, text in pages:
        if fix_s:                  # twice: "acce−−" needs the first "s" before the second
            text = _BROKEN_S.sub("s", _BROKEN_S.sub("s", text))
        counts["ligatures"] += len(re.findall("[\ufb00-\ufb06]", text))
        text = text.translate(_LIGATURES)
        if join_capitals:
            counts["split_words"] += sum(1 for m in _SPLIT_CAPITAL.finditer(text) if joinable(m))
            text = _SPLIT_CAPITAL.sub(lambda m: m.group(1) + m.group(2) if joinable(m) else m.group(0), text)
        text, n = _SPLIT_HEADING.subn(r"\1\2\3", text)
        counts["split_words"] += n
        text, n = _WATERMARK.subn(" ", text)
        counts["watermarks"] += n
        for owner in owners:
            text, n = owner.subn(" ", text)
            counts["watermarks"] += n
        text, n = _STAMP.subn(" ", text)
        counts["stamps"] += n
        out.append((page, text))
    if report is not None:
        report.update(repaired_s=broken if fix_s else 0, **counts)
    return out


def extract_pages(path: str, report: dict | None = None) -> Pages:
    """Pages of text. If `report` is a dict, it receives the number of repaired
    characters ("repaired_s") and removed watermarks ("watermarks")."""
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(f"Unsupported file type '{ext}' ({name}). "
                              f"Supported: {' '.join(sorted(SUPPORTED_EXTENSIONS))}")
    reader = {".pdf": _pdf, ".docx": _docx, ".epub": _epub,
              ".html": _html, ".htm": _html}.get(ext, _plain)
    try:
        return repair(reader(path), report)
    except ExtractionError:
        raise
    except Exception as e:            # corrupt or encrypted files, parser bugs
        raise ExtractionError(f"Could not read {name}: {e}") from e


def extract_text(path: str) -> str:
    return "\n\n".join(text for _, text in extract_pages(path))


def _need(module: str, package: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        raise ExtractionError(f"Missing {package}. Run:  pip install {package}") from None


def _plain(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [(None, f.read())]


def _pdf(path):
    pypdf = _need("pypdf", "pypdf")
    reader = pypdf.PdfReader(path)
    pages, errors = [], []
    for i, page in enumerate(reader.pages, 1):
        try:
            pages.append((i, page.extract_text() or ""))
        except Exception as e:            # one damaged page shouldn't lose the whole book
            errors.append(e)
            pages.append((i, ""))
    if errors and len(errors) == len(pages):
        raise errors[0]                    # nothing readable: report why (encrypted, damaged)
    return pages


def _docx(path):
    docx = _need("docx", "python-docx")
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = docx.Document(path)
    parts = []
    # Walk the body in order so tables stay next to the text around them.
    for el in doc.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            parts.append(Paragraph(el, doc).text)
        elif tag == "tbl":
            for row in Table(el, doc).rows:
                cells = []
                for cell in row.cells:
                    if not cells or cell.text != cells[-1]:      # merged cells repeat
                        cells.append(cell.text)
                parts.append(" | ".join(cells))
    return [(None, "\n".join(parts))]


def _soup_text(markup) -> str:
    bs4 = _need("bs4", "beautifulsoup4")
    soup = bs4.BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _html(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        markup = f.read()
    try:
        return [(None, _soup_text(markup))]
    except ExtractionError:
        markup = re.sub(r"(?is)<(script|style).*?</\1>", " ", markup)
        return [(None, re.sub(r"<[^>]+>", " ", markup))]


def _epub(path):
    ebooklib = _need("ebooklib", "ebooklib")
    from ebooklib import epub
    book = epub.read_epub(path)
    parts = [_soup_text(item.get_body_content())
             for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
    return [(None, "\n\n".join(parts))]
