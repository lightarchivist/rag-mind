"""
RAG-MIND Web — FastAPI backend
Lightweight RAG using a numpy index (no ChromaDB), sharing its pipeline with
the CLI through the `ragmind` package. Background indexing with progress.
"""

import json
import os
import shutil
import sys
import threading
from typing import Iterator

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import ragmind  # noqa: F401  (installed next to this file in Docker)
except ImportError:                       # running from a git checkout: use the repo copy
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ragmind import pipeline, thinking
from ragmind.chunking import chunk_pages
from ragmind.config import Settings, load_settings
from ragmind.extract import SUPPORTED_EXTENSIONS, ExtractionError, extract_pages
from ragmind.numpy_index import NumpyIndex
from ragmind.ollama import ModelNotFound, NotAChatModel, OllamaClient, OllamaError, short
from ragmind.prompt import cite

DATA_DIR   = os.getenv("DATA_DIR", "/data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_DIR  = DATA_DIR                    # index.npz + index.json (was index.pkl)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

settings = load_settings(Settings(ollama_base="http://host.docker.internal:11434"),
                         config_path=os.getenv("RAGMIND_CONFIG") or os.path.join(DATA_DIR, "config.json"))
ollama = OllamaClient(settings.ollama_base, timeout=settings.timeout)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Index (kept in memory, saved after every change) ─────────────────────────
index_lock = threading.RLock()
_index: NumpyIndex | None = None
indexing_status: dict[str, dict] = {}
rebuild_status = {"status": "idle", "done": 0, "total": 0, "error": ""}


def get_index() -> NumpyIndex:
    global _index
    with index_lock:
        if _index is None:
            _index = NumpyIndex.load(INDEX_DIR)
        return _index


def index_problem() -> str | None:
    idx = get_index()
    if len(idx) == 0:
        return None
    return pipeline.signature_problem(idx.signature, settings)


def _chunk_meta(name, ch):
    meta = {"source": name, "chunk_id": ch.chunk_id}
    if ch.page is not None:
        meta["page"] = ch.page
    return meta


# ─── Ollama errors → HTTP ─────────────────────────────────────────────────────
def http_error(e: OllamaError) -> HTTPException:
    if isinstance(e, (ModelNotFound, NotAChatModel)):
        return HTTPException(400, str(e))
    return HTTPException(503, str(e))


# ─── Upload paths ─────────────────────────────────────────────────────────────
def safe_filename(name: str) -> str:
    """Reduce a client-supplied name to a plain file name (no folders, no '..',
    no control or Windows-reserved characters)."""
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '<>:"|?*')
    name = name.strip().lstrip(".").strip()
    if not name or len(name) > 200:
        raise HTTPException(400, "Invalid file name")
    return name


def upload_path(name: str) -> str:
    """Full path for a file in UPLOAD_DIR; refuses anything that would land elsewhere."""
    path = os.path.join(UPLOAD_DIR, safe_filename(name))
    if os.path.dirname(os.path.realpath(path)) != os.path.realpath(UPLOAD_DIR):
        raise HTTPException(400, "Invalid file name")
    return path


# ─── Background indexing ──────────────────────────────────────────────────────
def index_file_background(path: str):
    name = os.path.basename(path)
    try:
        indexing_status[name] = {"status": "indexing", "progress": 0, "chunks": 0}
        why = index_problem()
        if why:
            raise ValueError(f"The index needs rebuilding first ({why}).")
        if name in get_index().sources():
            indexing_status[name] = {"status": "done", "progress": 100,
                                     "chunks": get_index().sources()[name]}
            return

        chunks = chunk_pages(extract_pages(path), settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("No text found (scanned PDFs need OCR first)")

        def progress(done, total):
            indexing_status[name] = {"status": "indexing", "progress": int(done / total * 100), "chunks": done}

        # Embed everything first; the index only changes once all of it worked.
        vectors = pipeline.embed_documents(ollama, settings, [ch.text for ch in chunks], progress)
        with index_lock:
            idx = get_index()
            if len(idx) == 0:
                idx.signature = pipeline.index_signature(settings)
            idx.delete_source(name)
            idx.add(vectors, [ch.text for ch in chunks], [_chunk_meta(name, ch) for ch in chunks])
            idx.save(INDEX_DIR)
        indexing_status[name] = {"status": "done", "progress": 100, "chunks": len(chunks)}
    except (ExtractionError, OllamaError, ValueError, OSError) as e:
        indexing_status[name] = {"status": "error", "progress": 0, "chunks": 0, "error": str(e)}
    except Exception as e:                     # keep the server up; report the problem
        import traceback; traceback.print_exc()
        indexing_status[name] = {"status": "error", "progress": 0, "chunks": 0, "error": str(e)}


def rebuild_background():
    """Re-embed every indexed document with the current settings. Files still in
    the upload folder are re-read (gaining page numbers); others keep their
    stored text. The live index is replaced only when everything succeeded."""
    global _index
    try:
        old = get_index()
        sources = sorted(old.sources())
        rebuild_status.update(status="rebuilding", done=0, total=len(sources), error="")
        new = NumpyIndex(pipeline.index_signature(settings))
        for name in sources:
            path = os.path.join(UPLOAD_DIR, name)
            if os.path.isfile(path):
                chunks = chunk_pages(extract_pages(path), settings.chunk_size, settings.chunk_overlap)
                texts, metas = [ch.text for ch in chunks], [_chunk_meta(name, ch) for ch in chunks]
            else:
                rows = old.chunks_of(name)
                texts = [d for d, _ in rows]
                metas = [dict(m, chunk_id=i) for i, (_, m) in enumerate(rows)]
            new.add(pipeline.embed_documents(ollama, settings, texts), texts, metas)
            rebuild_status["done"] += 1
        with index_lock:
            new.save(INDEX_DIR)
            _index = new
        rebuild_status["status"] = "done"
    except Exception as e:
        rebuild_status.update(status="error", error=str(e))


# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG-MIND", version="2.0")


class ChatRequest(BaseModel):
    question: str
    history: list = []
    filter_source: str = ""
    model: str = ""


# Endpoints that touch the disk or Ollama are plain `def`, so FastAPI runs them
# in a worker thread instead of blocking the event loop for every other user.

@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    dest = upload_path(file.filename)
    name = os.path.basename(dest)
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type: {ext}")
    if indexing_status.get(name, {}).get("status") == "indexing":
        raise HTTPException(409, f"{name} is being indexed right now; try again when it finishes")
    refuse_during_rebuild()
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # A re-uploaded file may have new content, so drop its old chunks;
    # the user then re-indexes it from the Library.
    with index_lock:
        idx = get_index()
        replaced = idx.delete_source(name) > 0
        if replaced:
            idx.save(INDEX_DIR)
    indexing_status.pop(name, None)
    return {"filename": name, "size": os.path.getsize(dest), "status": "uploaded", "replaced": replaced}


@app.get("/files")
def list_files():
    indexed = get_index().sources()
    files = []
    for fname in sorted(os.listdir(UPLOAD_DIR)):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            status = indexing_status.get(fname, {})
            files.append({
                "name": fname,
                "size": os.path.getsize(fpath),
                "indexed": fname in indexed,
                "chunks": indexed.get(fname, 0),
                "indexing_status": status.get("status", "idle"),
                "indexing_progress": status.get("progress", 0),
                "error": status.get("error", ""),
            })
    return {"files": files}


@app.post("/index/{filename}")
def index_document(filename: str, background_tasks: BackgroundTasks):
    fpath = upload_path(filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, f"File not found: {filename}")
    refuse_during_rebuild()
    name = os.path.basename(fpath)
    if indexing_status.get(name, {}).get("status") == "indexing":
        return {"filename": name, "status": "already_indexing"}
    # Mark it now, so a quick second click can't start a second indexing job.
    indexing_status[name] = {"status": "indexing", "progress": 0, "chunks": 0}
    background_tasks.add_task(index_file_background, fpath)
    return {"filename": name, "status": "started"}


@app.get("/index/status/{filename}")
def index_status(filename: str):
    status = indexing_status.get(filename, {"status": "idle", "progress": 0, "chunks": 0})
    return {"filename": filename, **status}


@app.post("/reindex")
def reindex(background_tasks: BackgroundTasks):
    if rebuild_status["status"] == "rebuilding":
        return {"status": "already_rebuilding"}
    if any(s.get("status") == "indexing" for s in indexing_status.values()):
        raise HTTPException(409, "A file is being indexed; try again when it finishes")
    rebuild_status.update(status="rebuilding", done=0, total=len(get_index().sources()), error="")
    background_tasks.add_task(rebuild_background)
    return {"status": "started"}


@app.get("/reindex/status")
def reindex_status():
    return dict(rebuild_status)


@app.get("/documents")
def list_documents():
    sources = get_index().sources()
    return {"documents": [{"name": k, "chunks": v} for k, v in sorted(sources.items())]}


def refuse_during_rebuild():
    if rebuild_status["status"] == "rebuilding":
        raise HTTPException(409, "The index is being rebuilt; try again when it finishes")


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    refuse_during_rebuild()
    with index_lock:
        idx = get_index()
        if not idx.delete_source(filename):
            raise HTTPException(404, "Document not found in index")
        idx.save(INDEX_DIR)
    return {"deleted": filename}


@app.delete("/files/{filename}")
def delete_file(filename: str):
    refuse_during_rebuild()
    fpath = upload_path(filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, "File not found")
    name = os.path.basename(fpath)
    with index_lock:
        idx = get_index()
        if idx.delete_source(name):
            idx.save(INDEX_DIR)
    os.remove(fpath)
    indexing_status.pop(name, None)
    return {"deleted": name}


@app.get("/models")
def list_models():
    """Installed chat models (embedding models are left out)."""
    try:
        models = ollama.chat_models()
    except OllamaError as e:
        raise http_error(e)
    return {"models": [m.to_dict() for m in models], "default": short(settings.llm_model)}


class IndexCorpus:
    """The web index seen as a set of documents (for broad questions)."""

    def keys(self):
        with index_lock:
            return sorted(get_index().sources())

    def search(self, vector, n, key=None):
        with index_lock:
            return get_index().query(vector, n, source_filter=key or "")

    def chunk_count(self, key):
        with index_lock:
            return get_index().sources().get(key, 0)

    def chunk_range(self, key, start, stop):
        with index_lock:
            rows = get_index().chunks_of(key)[start:stop]
        return [{"text": text, "source": m["source"], "page": m.get("page"), "score": None,
                 "chunk_id": m.get("chunk_id")} for text, m in rows]


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if len(get_index()) == 0:
        return StreamingResponse(iter(["No documents indexed yet. Upload a file and click Index."]),
                                 media_type="text/plain")
    why = index_problem()
    if why:
        return StreamingResponse(iter([f"The index needs rebuilding ({why}). "
                                       "Click “Rebuild index” at the top of the page."]),
                                 media_type="text/plain")
    try:
        model = ollama.resolve_chat_model(req.model or settings.llm_model)

        def search(vector, n):
            with index_lock:
                return get_index().query(vector, n, source_filter=req.filter_source)

        history = [h for h in req.history if isinstance(h, dict)]
        # Broad questions ("compare the documents") draw on every document,
        # unless the user limited the chat to one document.
        corpus = None if req.filter_source else IndexCorpus()
        prepared = pipeline.prepare(ollama, settings, model.name, req.question, history, search,
                                    corpus=corpus)
    except OllamaError as e:
        raise http_error(e)

    header = {"sources": list(dict.fromkeys(cite(ch) for ch in prepared.chunks)),
              "model": short(model.name)}
    if prepared.search_query != req.question:
        header["search_query"] = prepared.search_query
    if prepared.broad:
        header["documents"] = len({ch["source"] for ch in prepared.chunks})
    hold = thinking.reasons_unseparated(model)
    if hold:
        header["thinking"] = True       # the page shows "thinking…" until the answer starts

    def generate() -> Iterator[str]:
        yield json.dumps(header) + "\n"
        try:
            tokens = ollama.generate_stream(model.name, prepared.prompt,
                                            pipeline.generation_options(prepared.num_ctx))
            yield from thinking.visible(tokens, hold=hold)
        except OllamaError as e:
            yield f"\n[ERROR: {e}]"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
def health():
    return {"ollama": ollama.ping(), "model": short(settings.llm_model),
            "needs_reindex": index_problem(), "rebuild": dict(rebuild_status)}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
