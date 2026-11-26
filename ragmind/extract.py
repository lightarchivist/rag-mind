"""Text extraction. Returns a list of (page_number, text) so answers can cite
pages; formats without pages return a single (None, text) entry."""

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
_WATERMARK = re.compile(r"From the Library of [^\n]{1,80}?\s*ptg\d{6,}|\bptg\d{6,}\b")


# Typographic ligatures: "efﬁciency" contains one character "ﬁ", so it would
# not match "efficiency" in a search.
_LIGATURES = str.maketrans({"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
                            "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"})
# Download stamps on every page of papers from IEEE Xplore, naming the
# institution: "Authorized licensed use limited to: … Restrictions apply."
_STAMP = re.compile(r"Authorized licensed use limited to:[^\n]{0,250}?Restrictions apply\.?")
# Letter spacing that splits a capital from the rest of its word ("T able 1",
# "T o ensure", "V ery"), common in some journals' PDFs. Joined only where it
# happens throughout a document, since "T cells" is a real phrase elsewhere.
_SPLIT_CAPITAL = re.compile(r"\b([TVWY]) ([a-z]+)\b")
SPLIT_CAPITAL_MIN = 10
# Small capitals in numbered headings come out as "VIII. C ONCLUSION".
_SPLIT_HEADING = re.compile(r"(\b[IVX]+\.\s+)([A-Z]) ([A-Z]{2,})\b")


def repair(pages: Pages, report: dict | None = None) -> Pages:
    """Undo common PDF text problems: the broken-"s" font, ligatures, split
    capitals and small-capital headings; remove eBook watermarks and download
    stamps. `report` receives the counts."""
    whole = "".join(text for _, text in pages)
    broken = len(_BROKEN_S.findall(whole)) + len(re.findall(r"(?<=[A-Za-z]\u2212)\u2212(?![a-z])", whole))
    fix_s = broken >= BROKEN_S_MIN and broken >= 0.5 * whole.count("s")
    join_capitals = len(_SPLIT_CAPITAL.findall(whole)) >= SPLIT_CAPITAL_MIN
    counts = {"watermarks": 0, "stamps": 0, "ligatures": 0, "split_words": 0}
    out = []
    for page, text in pages:
        if fix_s:                  # twice: "acce−−" needs the first "s" before the second
            text = _BROKEN_S.sub("s", _BROKEN_S.sub("s", text))
        counts["ligatures"] += len(re.findall("[\ufb00-\ufb06]", text))
        text = text.translate(_LIGATURES)
        if join_capitals:
            text, n = _SPLIT_CAPITAL.subn(r"\1\2", text)
            counts["split_words"] += n
        text, n = _SPLIT_HEADING.subn(r"\1\2\3", text)
        counts["split_words"] += n
        text, n = _WATERMARK.subn(" ", text)
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
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]


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
