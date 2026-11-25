"""Text extraction. Returns a list of (page_number, text) so answers can cite
pages; formats without pages return a single (None, text) entry."""

import os
import re

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".rst", ".html", ".htm", ".epub"}

Pages = list[tuple[int | None, str]]


class ExtractionError(Exception):
    """A file could not be read. The message is meant for the user."""


def extract_pages(path: str) -> Pages:
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(f"Unsupported file type '{ext}' ({name}). "
                              f"Supported: {' '.join(sorted(SUPPORTED_EXTENSIONS))}")
    reader = {".pdf": _pdf, ".docx": _docx, ".epub": _epub,
              ".html": _html, ".htm": _html}.get(ext, _plain)
    try:
        return reader(path)
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
