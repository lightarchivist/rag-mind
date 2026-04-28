"""
RagMind Web — FastAPI backend
Exposes the RAG pipeline over HTTP for the web UI.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import os, json, hashlib, shutil, asyncio
from typing import AsyncGenerator

# ─── Re-use core RAG logic ────────────────────────────────────────────────────
import sys
sys.path.insert(0, "/app")

OLLAMA_BASE  = os.getenv("OLLAMA_BASE", "http://host.docker.internal:11434")
LLM_MODEL    = os.getenv("LLM_MODEL",   "llama3")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR   = os.getenv("CHROMA_DIR",  "/data/chromadb")
UPLOAD_DIR   = os.getenv("UPLOAD_DIR",  "/data/uploads")
COLLECTION   = "rag_documents"
CHUNK_SIZE   = 500
CHUNK_OVERLAP= 80
TOP_K        = 6

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

import urllib.request, urllib.error

# ─── Ollama helpers ───────────────────────────────────────────────────────────
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
        for sep in (". ", ".\n", "? ", "! ", "\n\n"):
            pos = text.rfind(sep, start + CHUNK_SIZE // 2, end)
            if pos != -1:
                end = pos + len(sep); break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": os.path.basename(source), "chunk_id": idx})
            idx += 1
        start = end - CHUNK_OVERLAP
    return chunks

# ─── Vector store ─────────────────────────────────────────────────────────────
def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"})

def index_file(path):
    text = extract_text(path)
    if not text.strip():
        raise ValueError("Empty or unreadable file")
    chunks = chunk_text(text, path)
    col = get_collection()
    existing = col.get(where={"source": os.path.basename(path)})
    if existing["ids"]:
        return {"status": "skipped", "chunks": len(existing["ids"])}
    file_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    ids, embeddings, documents, metadatas = [], [], [], []
    for i, ch in enumerate(chunks):
        ids.append(f"{file_hash}_{i}")
        embeddings.append(embed(ch["text"]))
        documents.append(ch["text"])
        metadatas.append({"source": ch["source"], "chunk_id": ch["chunk_id"]})
    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return {"status": "indexed", "chunks": len(chunks)}

def retrieve(question, top_k=TOP_K):
    col = get_collection()
    if col.count() == 0:
        return []
    q_emb = embed(question)
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"])
    return [
        {"text": doc, "source": meta["source"], "score": round(1 - dist, 3)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0])
    ]

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="RagMind", version="1.0")

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

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("/app/static/index.html") as f:
        return f.read()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = {".pdf", ".epub", ".docx", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported type: {ext}")
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = index_file(dest)
        return {"filename": file.filename, **result}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/documents")
async def list_documents():
    col = get_collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    sources = {}
    for m in all_meta:
        sources[m["source"]] = sources.get(m["source"], 0) + 1
    return {"documents": [{"name": k, "chunks": v} for k, v in sorted(sources.items())]}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    col = get_collection()
    existing = col.get(where={"source": filename})
    if not existing["ids"]:
        raise HTTPException(404, "Document not found")
    col.delete(ids=existing["ids"])
    # remove uploaded file
    fpath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(fpath):
        os.remove(fpath)
    return {"deleted": filename}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    chunks = retrieve(req.question)
    if not chunks:
        async def no_docs():
            yield "No documents indexed yet. Please upload some documents first."
        return StreamingResponse(no_docs(), media_type="text/plain")

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

@app.get("/health")
async def health():
    try:
        _post("/api/tags", {})
        ollama_ok = True
    except:
        ollama_ok = False
    return {"ollama": ollama_ok, "model": LLM_MODEL}

app.mount("/static", StaticFiles(directory="/app/static"), name="static")
