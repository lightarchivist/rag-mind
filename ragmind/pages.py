"""PDF pages as images, so the figures and diagrams next to a passage can be seen.

Text extraction leaves pictures out. Rather than have a model describe them
(slow, and it can misread labels), the web app shows the page itself: nothing
is generated or guessed. Pages are drawn with PDFium (pypdfium2) and cached.
"""
import hashlib
import os
import struct
import threading
import zlib

DEFAULT_WIDTH = 1000          # pixels; the page fits the chat column
MIN_WIDTH, MAX_WIDTH = 300, 2400

_lock = threading.Lock()      # PDFium must not be used from two threads at once


class PageError(Exception):
    """The page doesn't exist or the file can't be drawn."""


class RendererMissing(PageError):
    """pypdfium2 is not installed."""


def _pdfium():
    try:
        import pypdfium2
    except ImportError:
        raise RendererMissing("Page images need the pypdfium2 package: pip install pypdfium2") from None
    return pypdfium2


def png_bytes(pixels) -> bytes:
    """Encode an RGB or RGBA image (height x width x 3/4 uint8 array) as PNG."""
    height, width, channels = pixels.shape
    colour_type = {3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(height))   # filter 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data +
                struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def render(path: str, page: int, width: int = DEFAULT_WIDTH) -> bytes:
    """PNG of one page (numbered from 1), `width` pixels wide."""
    pdfium = _pdfium()
    width = max(MIN_WIDTH, min(MAX_WIDTH, int(width)))
    with _lock:
        try:
            pdf = pdfium.PdfDocument(path)
        except Exception as e:                       # encrypted, damaged, not a PDF
            raise PageError(f"Could not open {os.path.basename(path)}: {e}") from None
        try:
            if not 1 <= page <= len(pdf):
                raise PageError(f"{os.path.basename(path)} has no page {page} (it has {len(pdf)})")
            pg = pdf[page - 1]
            try:
                bitmap = pg.render(scale=width / pg.get_width(), rev_byteorder=True)
                pixels = bitmap.to_numpy()
            finally:
                pg.close()
        finally:
            pdf.close()
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise PageError("Unexpected page image format")
    return png_bytes(pixels)


def cache_folder(cache_dir: str, name: str) -> str:
    """Where a file's page images are kept (one folder per file name)."""
    return os.path.join(cache_dir, hashlib.sha1(name.encode("utf-8")).hexdigest()[:16])


def page_png(path: str, page: int, width: int, cache_dir: str) -> bytes:
    """A page image, from the cache when the file hasn't changed since it was drawn."""
    width = max(MIN_WIDTH, min(MAX_WIDTH, int(width)))
    st = os.stat(path)
    folder = cache_folder(cache_dir, os.path.basename(path))
    cached = os.path.join(folder, f"{st.st_mtime_ns}-{st.st_size}-{page}-{width}.png")
    if os.path.isfile(cached):
        with open(cached, "rb") as f:
            return f.read()
    png = render(path, page, width)
    try:
        os.makedirs(folder, exist_ok=True)
        current = f"{st.st_mtime_ns}-{st.st_size}-"
        for old in os.listdir(folder):              # images of an earlier version of the file
            if not old.startswith(current):
                os.remove(os.path.join(folder, old))
        tmp = cached + ".tmp"
        with open(tmp, "wb") as f:
            f.write(png)
        os.replace(tmp, cached)
    except OSError:
        pass                                         # caching is optional
    return png
