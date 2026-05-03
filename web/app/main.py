"""
RAG-MIND Web — FastAPI backend
Lightweight RAG using numpy, no ChromaDB.
Background indexing with progress tracking.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import os, json, hashlib, shutil, pickle, time, threading
from typing import AsyncGenerator
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
indexing_status = {}  # filename -> {"status": "indexing|done|error", "progress": 0-100, "chunks": 0}
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
        # filter by source if specified
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
    resp = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
    return resp["embedding"]

async def stream_ollama(prompt: str) -> AsyncGenerator[str, None]:
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
    except urllib.error.URLError as e:
        yield f"\n[ERROR: {e}]"

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

# ─── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text, source):
    text = " ".join(text.split())
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        for sep in (". ", "? ", "! ", "\n\n"):
            pos = text.rfind(sep, start + CHUNK_SIZE // 2, end)
            if pos != -1:
                end = pos + len(sep); break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": os.path.basename(source), "chunk_id": idx})
            idx += 1
        start = end - CHUNK_OVERLAP
    return chunks

# ─── Background indexing ──────────────────────────────────────────────────────
def index_file_background(path: str):
    filename = os.path.basename(path)
    try:
        indexing_status[filename] = {"status": "indexing", "progress": 0, "chunks": 0}

        text = extract_text(path)
        if not text.strip():
            indexing_status[filename] = {"status": "error", "progress": 0, "chunks": 0, "error": "Empty file"}
            return

        with index_lock:
            idx = load_index()
            if filename in idx.sources():
                indexing_status[filename] = {"status": "done", "progress": 100, "chunks": idx.sources()[filename]}
                return

        chunks = chunk_text(text, path)
        total = len(chunks)
        batch_emb, batch_doc, batch_meta = [], [], []

        for i, ch in enumerate(chunks):
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
                import gc; gc.collect()
                time.sleep(0.1)

        if batch_emb:
            with index_lock:
                idx = load_index()
                idx.add(batch_emb, batch_doc, batch_meta)
                save_index(idx)

        indexing_status[filename] = {"status": "done", "progress": 100, "chunks": total}

    except Exception as e:
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
    filter_source: str = ""  # empty = all documents

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return f.read()

# ── Upload file ───────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = {".pdf", ".epub", ".docx", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported type: {ext}")
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    size = os.path.getsize(dest)
    return {"filename": file.filename, "size": size, "status": "uploaded"}

# ── List uploaded files ───────────────────────────────────────────────────────
@app.get("/files")
async def list_files():
    with index_lock:
        idx = load_index()
    indexed = idx.sources()
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

# ── Start indexing in background ──────────────────────────────────────────────
@app.post("/index/{filename}")
async def index_document(filename: str, background_tasks: BackgroundTasks):
    fpath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, f"File not found: {filename}")
    if indexing_status.get(filename, {}).get("status") == "indexing":
        return {"filename": filename, "status": "already_indexing"}
    background_tasks.add_task(index_file_background, fpath)
    return {"filename": filename, "status": "started"}

# ── Indexing progress ─────────────────────────────────────────────────────────
@app.get("/index/status/{filename}")
async def index_status(filename: str):
    status = indexing_status.get(filename, {"status": "idle", "progress": 0, "chunks": 0})
    return {"filename": filename, **status}

# ── List indexed documents ────────────────────────────────────────────────────
@app.get("/documents")
async def list_documents():
    with index_lock:
        idx = load_index()
    sources = idx.sources()
    return {"documents": [{"name": k, "chunks": v} for k, v in sorted(sources.items())]}

# ── Remove from index ─────────────────────────────────────────────────────────
@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    with index_lock:
        idx = load_index()
        if filename not in idx.sources():
            raise HTTPException(404, "Document not found in index")
        idx.delete_source(filename)
        save_index(idx)
    return {"deleted": filename}

# ── Delete file entirely ──────────────────────────────────────────────────────
@app.delete("/files/{filename}")
async def delete_file(filename: str):
    fpath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, "File not found")
    with index_lock:
        idx = load_index()
        if filename in idx.sources():
            idx.delete_source(filename)
            save_index(idx)
    os.remove(fpath)
    indexing_status.pop(filename, None)
    return {"deleted": filename}

# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    with index_lock:
        idx = load_index()
    if not idx.embeddings:
        async def no_docs():
            yield "No documents indexed yet. Upload a file and click Index."
        return StreamingResponse(no_docs(), media_type="text/plain")

    q_emb = embed(req.question)
    chunks = idx.query(q_emb, n=TOP_K, source_filter=req.filter_source)

    context = "\n\n".join(
        f"[{i+1}] (source: {ch['source']}, relevance: {ch['score']})\n{ch['text']}"
        for i, ch in enumerate(chunks))

    history_text = ""
    for h in req.history[-4:]:
        history_text += f"User: {h['q']}\nAssistant: {h['a']}\n"

    prompt = f"{SYSTEM}\n\n=== RETRIEVED PASSAGES ===\n{context}\n=== END ===\n\n{history_text}User: {req.question}\nAssistant:"
    sources = list({ch["source"] for ch in chunks})

    async def generate():
        yield json.dumps({"sources": sources}) + "\n"
        async for token in stream_ollama(prompt):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        _post("/api/tags", {})
        ollama_ok = True
    except:
        ollama_ok = False
    return {"ollama": ollama_ok, "model": LLM_MODEL}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")