"""
RAG-MIND Web — FastAPI backend
Lightweight RAG using numpy, no ChromaDB.
Background indexing with progress tracking and gc.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import os, json, shutil, pickle, time, threading, gc
from typing import Iterator
import numpy as np

OLLAMA_BASE   = os.getenv("OLLAMA_BASE", "http://host.docker.internal:11434")
LLM_MODEL     = os.getenv("LLM_MODEL",   "llama3")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "nomic-embed-text")
DATA_DIR      = os.getenv("DATA_DIR",    "/data")
UPLOAD_DIR    = os.path.join(DATA_DIR, "uploads")
INDEX_FILE    = os.path.join(DATA_DIR, "index.pkl")
STATIC_DIR    = os.path.join(os.path.dirname(__file__), "static")
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
TOP_K         = 6
BATCH_SIZE    = 3

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

import urllib.request, urllib.error

# ─── Indexing progress tracker ────────────────────────────────────────────────
indexing_status = {}
index_lock = threading.Lock()

# ─── Simple numpy vector index ────────────────────────────────────────────────
class VectorIndex:
    def __init__(self):
        self.embeddings = []
        self.documents  = []
        self.metadatas  = []

    def add(self, embeddings, documents, metadatas):
        self.embeddings.extend([np.array(e) for e in embeddings])
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def query(self, query_embedding, n=TOP_K, source_filter=""):
        if not self.embeddings:
            return []
        q = np.array(query_embedding)
        if source_filter:
            indices = [i for i, m in enumerate(self.metadatas) if m["source"] == source_filter]
            if not indices:
                return []
            matrix = np.stack([self.embeddings[i] for i in indices])
            norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
            norms = np.where(norms == 0, 1e-10, norms)
            scores = matrix.dot(q) / norms
            top_idx = np.argsort(scores)[::-1][:n]
            return [
                {"text": self.documents[indices[i]], "source": self.metadatas[indices[i]]["source"], "score": round(float(scores[i]), 3)}
                for i in top_idx
            ]
        matrix = np.stack(self.embeddings)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)
        scores = matrix.dot(q) / norms
        top_idx = np.argsort(scores)[::-1][:n]
        return [
            {"text": self.documents[i], "source": self.metadatas[i]["source"], "score": round(float(scores[i]), 3)}
            for i in top_idx
        ]

    def sources(self):
        counts = {}
        for m in self.metadatas:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def delete_source(self, source):
        keep = [i for i, m in enumerate(self.metadatas) if m["source"] != source]
        self.embeddings = [self.embeddings[i] for i in keep]
        self.documents  = [self.documents[i]  for i in keep]
        self.metadatas  = [self.metadatas[i]  for i in keep]

def load_index() -> VectorIndex:
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return VectorIndex()

def save_index(idx: VectorIndex):
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(idx, f)

# ─── Ollama ───────────────────────────────────────────────────────────────────
def _post(endpoint, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}{endpoint}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}")

def embed(text):
    data = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp["embedding"]

# A plain (sync) generator: StreamingResponse runs it in a worker thread, so the
# blocking urllib read doesn't freeze the server for everyone else.
def stream_ollama(prompt: str) -> Iterator[str]:
    data = json.dumps({
        "model": LLM_MODEL, "prompt": prompt, "stream": True,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                line = line.strip()
                if not line: continue
                chunk = json.loads(line)
                tok = chunk.get("response", "")
                if tok:
                    yield tok
                if chunk.get("done"):
                    break
    except OSError as e:          # URLError, timeouts, dropped connections
        yield f"\n[ERROR: {e}]"

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

# ─── Text extraction ──────────────────────────────────────────────────────────
def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        r = PdfReader(path)
        return "\n\n".join(p.extract_text() or "" for p in r.pages)
    elif ext == ".docx":
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif ext == ".epub":
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup as BS
        book = epub.read_epub(path)
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BS(item.get_body_content(), "html.parser")
            parts.append(soup.get_text())
        return "\n\n".join(parts)
    else:
        raise ValueError(f"Unsupported: {ext}")

# ─── Chunking (infinite loop safe) ───────────────────────────────────────────
def chunk_text(text, source):
    text = " ".join(text.split())
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in (". ", "? ", "! ", "\n\n"):
                pos = text.rfind(sep, start + CHUNK_SIZE // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": os.path.basename(source), "chunk_id": idx})
            idx += 1
        if end >= len(text):
            break                     # otherwise the tail gets indexed again as an extra chunk
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = start + CHUNK_SIZE
        start = next_start
    return chunks

# ─── Background indexing ──────────────────────────────────────────────────────
def index_file_background(path: str):
    filename = os.path.basename(path)
    try:
        indexing_status[filename] = {"status": "indexing", "progress": 0, "chunks": 0}
        print(f"DEBUG: Starting indexing {filename}", flush=True)

        text = extract_text(path)
        print(f"DEBUG: Extracted {len(text)} chars", flush=True)

        if not text.strip():
            indexing_status[filename] = {"status": "error", "progress": 0, "chunks": 0, "error": "Empty file"}
            return

        # check if already indexed without lock
        idx = load_index()
        already_indexed = filename in idx.sources()
        chunks_count = idx.sources().get(filename, 0)
        del idx
        gc.collect()

        if already_indexed:
            indexing_status[filename] = {"status": "done", "progress": 100, "chunks": chunks_count}
            print(f"DEBUG: Already indexed, skipping", flush=True)
            return

        print(f"DEBUG: Chunking text...", flush=True)
        chunks = chunk_text(text, path)
        del text
        gc.collect()
        print(f"DEBUG: Got {len(chunks)} chunks", flush=True)
        total = len(chunks)
        batch_emb, batch_doc, batch_meta = [], [], []

        for i, ch in enumerate(chunks):
            print(f"DEBUG: Embedding chunk {i}/{total}", flush=True)
            batch_emb.append(embed(ch["text"]))
            batch_doc.append(ch["text"])
            batch_meta.append({"source": ch["source"], "chunk_id": ch["chunk_id"]})

            progress = int((i + 1) / total * 100)
            indexing_status[filename] = {"status": "indexing", "progress": progress, "chunks": i + 1}

            if (i + 1) % BATCH_SIZE == 0:
                with index_lock:
                    idx = load_index()
                    idx.add(batch_emb, batch_doc, batch_meta)
                    save_index(idx)
                    del idx
                batch_emb, batch_doc, batch_meta = [], [], []
                gc.collect()
                time.sleep(0.1)

        if batch_emb:
            with index_lock:
                idx = load_index()
                idx.add(batch_emb, batch_doc, batch_meta)
                save_index(idx)
                del idx
            gc.collect()

        indexing_status[filename] = {"status": "done", "progress": 100, "chunks": total}
        print(f"DEBUG: Done indexing {filename} — {total} chunks", flush=True)

    except Exception as e:
        print(f"DEBUG: ERROR — {e}", flush=True)
        import traceback; traceback.print_exc()
        indexing_status[filename] = {"status": "error", "progress": 0, "chunks": 0, "error": str(e)}

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG-MIND", version="1.0")

SYSTEM = """You are a document analysis assistant.
Answer questions ONLY using the retrieved passages below.
Rules:
1. Base your answer solely on the provided passages.
2. If the answer is not in the passages, say: "This information is not found in the indexed documents."
3. Cite the source filename when relevant.
4. Do not use outside knowledge."""

class ChatRequest(BaseModel):
    question: str
    history: list = []
    filter_source: str = ""

# Endpoints that touch the disk or Ollama are plain `def`, so FastAPI runs them
# in a worker thread instead of blocking the event loop for every other user.

@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed = {".pdf", ".epub", ".docx", ".txt", ".md"}
    dest = upload_path(file.filename)
    name = os.path.basename(dest)
    ext = os.path.splitext(name)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported type: {ext}")
    if indexing_status.get(name, {}).get("status") == "indexing":
        raise HTTPException(409, f"{name} is being indexed right now; try again when it finishes")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # A re-uploaded file may have new content, so drop its old chunks;
    # the user then re-indexes it from the Library.
    replaced = False
    with index_lock:
        idx = load_index()
        if name in idx.sources():
            idx.delete_source(name)
            save_index(idx)
            replaced = True
        del idx
    indexing_status.pop(name, None)
    size = os.path.getsize(dest)
    return {"filename": name, "size": size, "status": "uploaded", "replaced": replaced}

@app.get("/files")
def list_files():
    with index_lock:
        idx = load_index()
    indexed = idx.sources()
    del idx
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
            })
    return {"files": files}

@app.post("/index/{filename}")
def index_document(filename: str, background_tasks: BackgroundTasks):
    fpath = upload_path(filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, f"File not found: {filename}")
    name = os.path.basename(fpath)
    if indexing_status.get(name, {}).get("status") == "indexing":
        return {"filename": name, "status": "already_indexing"}
    # Mark it now, so a quick second click can't start a second indexing job.
    indexing_status[name] = {"status": "indexing", "progress": 0, "chunks": 0}
    background_tasks.add_task(index_file_background, fpath)
    return {"filename": name, "status": "started"}

@app.get("/index/status/{filename}")
async def index_status(filename: str):
    status = indexing_status.get(filename, {"status": "idle", "progress": 0, "chunks": 0})
    return {"filename": filename, **status}

@app.get("/documents")
def list_documents():
    with index_lock:
        idx = load_index()
    sources = idx.sources()
    del idx
    return {"documents": [{"name": k, "chunks": v} for k, v in sorted(sources.items())]}

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    with index_lock:
        idx = load_index()
        if filename not in idx.sources():
            raise HTTPException(404, "Document not found in index")
        idx.delete_source(filename)
        save_index(idx)
    return {"deleted": filename}

@app.delete("/files/{filename}")
def delete_file(filename: str):
    fpath = upload_path(filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, "File not found")
    name = os.path.basename(fpath)
    with index_lock:
        idx = load_index()
        if name in idx.sources():
            idx.delete_source(name)
            save_index(idx)
    os.remove(fpath)
    indexing_status.pop(name, None)
    return {"deleted": name}

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    with index_lock:
        idx = load_index()
    if not idx.embeddings:
        return StreamingResponse(iter(["No documents indexed yet. Upload a file and click Index."]),
                                 media_type="text/plain")

    try:
        q_emb = embed(req.question)
    except OSError as e:
        raise HTTPException(503, f"Cannot reach Ollama: {e}")
    chunks = idx.query(q_emb, n=TOP_K, source_filter=req.filter_source)
    del idx

    context = "\n\n".join(
        f"[{i+1}] (source: {ch['source']}, relevance: {ch['score']})\n{ch['text']}"
        for i, ch in enumerate(chunks))

    history_text = ""
    for h in req.history[-4:]:
        history_text += f"User: {h['q']}\nAssistant: {h['a']}\n"

    prompt = f"{SYSTEM}\n\n=== RETRIEVED PASSAGES ===\n{context}\n=== END ===\n\n{history_text}User: {req.question}\nAssistant:"
    sources = list({ch["source"] for ch in chunks})

    def generate():
        yield json.dumps({"sources": sources}) + "\n"
        yield from stream_ollama(prompt)

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/health")
def health():
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            ollama_ok = r.status == 200
    except Exception:
        ollama_ok = False
    return {"ollama": ollama_ok, "model": LLM_MODEL}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
