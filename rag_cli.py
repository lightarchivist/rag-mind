# RagMind — Local RAG-powered Document Q&A CLI
# rag_cli.py — Universal Document Q&A CLI
# Copyright (C) 2026  Seb Szmytka
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# See <https://www.gnu.org/licenses/> for details.
#!/usr/bin/env python3
"""
rag_cli.py — Universal RAG-powered CLI for Document Q&A
Scales from a single paper to multiple books using Ollama + ChromaDB.

Usage:
    # Index one or more documents, then chat
    python rag_cli.py add paper.pdf
    python rag_cli.py add book1.pdf book2.pdf notes.txt
    python rag_cli.py chat
    python rag_cli.py list
    python rag_cli.py clear

Requirements:
    pip install -r requirements.txt
    ollama pull llama3
    ollama pull nomic-embed-text   # for embeddings
"""

import argparse
import os
import sys
import json
import hashlib
import time
import textwrap

# ─── Colours ──────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[96m"
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; DIM = "\033[2m"

def c(col, text):
    return f"{col}{text}{RESET}" if sys.stdout.isatty() else text

# ─── Config ───────────────────────────────────────────────────────────────────
OLLAMA_BASE   = "http://localhost:11434"
LLM_MODEL     = "llama3"
EMBED_MODEL   = "nomic-embed-text"   # fast, small, great for RAG
CHROMA_DIR    = os.path.expanduser("~/.rag_cli_db")
COLLECTION    = "rag_documents"
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 80     # overlap between chunks
TOP_K         = 6      # chunks to retrieve per query

# ─── Ollama helpers ────────────────────────────────────────────────────────────
import urllib.request, urllib.error

def _post(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        print(c(RED, f"\nERROR: Cannot reach Ollama at {OLLAMA_BASE}"))
        print("  Start it with:  ollama serve")
        sys.exit(1)

def embed(text: str) -> list[float]:
    """Get embedding vector from Ollama."""
    resp = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
    return resp["embedding"]

def stream_generate(prompt: str) -> str:
    """Stream a generation from Ollama, return full text."""
    import urllib.request
    payload = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    full = []
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                tok = chunk.get("response", "")
                if tok:
                    print(tok, end="", flush=True)
                    full.append(tok)
                if chunk.get("done"):
                    break
    except urllib.error.URLError as e:
        print(c(RED, f"\nERROR: {e}"))
    print()
    return "".join(full)

# ─── Text extraction ────────────────────────────────────────────────────────────
def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            r = PdfReader(path)
            return "\n\n".join(p.extract_text() or "" for p in r.pages)
        except ImportError:
            sys.exit(c(RED, "Missing pypdf. Run:  pip install pypdf"))

    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            sys.exit(c(RED, "Missing python-docx. Run:  pip install python-docx"))

    elif ext in (".txt", ".md", ".rst"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif ext in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator="\n")
        except ImportError:
            # fallback: strip tags with regex
            import re
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            return re.sub(r"<[^>]+>", " ", text)

    elif ext == ".epub":
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup as BS
            book = epub.read_epub(path)
            parts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BS(item.get_body_content(), "html.parser")
                parts.append(soup.get_text())
            return "\n\n".join(parts)
        except ImportError:
            sys.exit(c(RED, "For EPUB support:  pip install ebooklib beautifulsoup4"))
    else:
        sys.exit(c(RED, f"Unsupported file type: {ext}  (use .pdf .docx .txt .md .html .epub)"))

# ─── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    text = " ".join(text.split())   # normalise whitespace
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        # try to end at a sentence boundary
        for sep in (". ", ".\n", "? ", "! ", "\n\n"):
            pos = text.rfind(sep, start + CHUNK_SIZE // 2, end)
            if pos != -1:
                end = pos + len(sep)
                break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": os.path.basename(source),
                "chunk_id": idx,
            })
            idx += 1
        start = end - CHUNK_OVERLAP
    return chunks

# ─── Vector store (ChromaDB) ────────────────────────────────────────────────────
def get_collection():
    try:
        import chromadb
    except ImportError:
        sys.exit(c(RED, "Missing chromadb. Run:  pip install chromadb"))
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

def add_document(path: str):
    """Extract, chunk, embed, and store a document."""
    if not os.path.isfile(path):
        print(c(RED, f"  File not found: {path}"))
        return

    print(c(DIM, f"  Reading {os.path.basename(path)}…"), end="", flush=True)
    text = extract_text(path)
    if not text.strip():
        print(c(RED, " empty or unreadable."))
        return
    print(c(GREEN, f" {len(text):,} chars"))

    chunks = chunk_text(text, path)
    print(c(DIM, f"  Chunking → {len(chunks)} passages…"), end="", flush=True)

    col = get_collection()
    # deduplicate by file hash to avoid re-indexing same file
    file_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    existing_ids = col.get(where={"source": os.path.basename(path)})
    if existing_ids["ids"]:
        print(c(YELLOW, f" already indexed ({len(existing_ids['ids'])} chunks). Skipping."))
        return

    print(c(DIM, " embedding"), end="", flush=True)
    ids, embeddings, documents, metadatas = [], [], [], []
    for i, ch in enumerate(chunks):
        uid = f"{file_hash}_{i}"
        ids.append(uid)
        embeddings.append(embed(ch["text"]))
        documents.append(ch["text"])
        metadatas.append({"source": ch["source"], "chunk_id": ch["chunk_id"]})
        if i % 10 == 0:
            print(".", end="", flush=True)

    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(c(GREEN, f" done. ({len(chunks)} chunks stored)"))

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Find the top-k most relevant chunks for a question."""
    col = get_collection()
    q_emb = embed(question)
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "source": meta["source"], "score": 1 - dist})
    return chunks

def list_documents():
    col = get_collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    sources = {}
    for m in all_meta:
        sources[m["source"]] = sources.get(m["source"], 0) + 1
    if not sources:
        print(c(DIM, "  No documents indexed yet. Use:  python rag_cli.py add <file>"))
        return
    print(c(CYAN, "\n── Indexed documents ──────────────────────"))
    for src, n in sorted(sources.items()):
        print(f"  {src:40s} {n:4d} chunks")
    print(c(CYAN, "────────────────────────────────────────────\n"))

def clear_collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    client.delete_collection(COLLECTION)
    print(c(YELLOW, "  All indexed documents cleared."))

# ─── RAG prompt ────────────────────────────────────────────────────────────────
SYSTEM = """You are a document analysis assistant.
Answer questions ONLY using the retrieved passages below.
Rules:
1. Base your answer solely on the provided passages.
2. If the answer is not in the passages, say: "This information is not found in the indexed documents."
3. Cite the source filename when relevant.
4. Do not use outside knowledge."""

def build_rag_prompt(question: str, chunks: list[dict], history: list) -> str:
    context_parts = []
    for i, ch in enumerate(chunks, 1):
        context_parts.append(f"[{i}] (source: {ch['source']}, relevance: {ch['score']:.2f})\n{ch['text']}")
    context = "\n\n".join(context_parts)

    history_text = ""
    if history:
        lines = []
        for h in history[-4:]:
            lines.append(f"User: {h['q']}\nAssistant: {h['a']}")
        history_text = "\n".join(lines) + "\n"

    return (
        f"{SYSTEM}\n\n"
        f"=== RETRIEVED PASSAGES ===\n{context}\n=== END PASSAGES ===\n\n"
        f"{history_text}"
        f"User: {question}\nAssistant:"
    )

# ─── Chat REPL ──────────────────────────────────────────────────────────────────
BANNER = r"""
  ____      _    ____     ____ _     ___
 |  _ \ __ _/ _\  / ___| / ___| |   |_ _|
 | |_) / _` | |_|| |   | |   | |    | |
 |  _ < (_| |__. | |___| |___| |___ | |
 |_| \_\__,_|___/  \____|\____|_____|___|

  Universal Document Q&A  •  RAG + Ollama + Llama 3
"""

HELP = """
Commands:
  <question>   Ask anything about indexed documents
  /sources     Show which documents are indexed
  /history     Show conversation history
  /clear       Clear conversation history
  /quit        Exit
"""

def chat_repl():
    col = get_collection()
    count = col.count()
    if count == 0:
        print(c(YELLOW, "No documents indexed. Add some first:\n  python rag_cli.py add paper.pdf"))
        return

    sources = {}
    for m in col.get(include=["metadatas"])["metadatas"]:
        sources[m["source"]] = sources.get(m["source"], 0) + 1

    print(c(CYAN, BANNER))
    print(c(BOLD, "  Indexed: ") + ", ".join(sources.keys()))
    print(c(BOLD, "  Chunks : ") + str(count))
    print(c(BOLD, "  Model  : ") + LLM_MODEL)
    print(c(DIM,  "  Type /help for commands, /quit to exit\n"))

    history = []
    while True:
        try:
            user_input = input(c(GREEN, "You › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(c(DIM, "\nGoodbye!"))
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print(c(DIM, "Goodbye!"))
            break
        elif user_input.lower() == "/help":
            print(c(CYAN, HELP))
            continue
        elif user_input.lower() in ("/sources", "/list"):
            list_documents()
            continue
        elif user_input.lower() == "/history":
            if not history:
                print(c(DIM, "  No history yet.\n"))
            for i, h in enumerate(history, 1):
                print(c(BOLD, f"\n  Q{i}: ") + h["q"])
                print(c(DIM,  f"  A{i}: ") + h["a"][:200] + ("…" if len(h["a"]) > 200 else ""))
            print()
            continue
        elif user_input.lower() == "/clear":
            history.clear()
            print(c(DIM, "  History cleared.\n"))
            continue

        # Retrieve relevant chunks
        chunks = retrieve(user_input)
        # Show source hint
        shown_sources = list({ch["source"] for ch in chunks})
        print(c(DIM, f"  [{', '.join(shown_sources)}] — {len(chunks)} passages retrieved"))

        prompt = build_rag_prompt(user_input, chunks, history)
        print(c(CYAN, "\nAssistant › "), end="", flush=True)
        t0 = time.time()
        answer = stream_generate(prompt)
        elapsed = time.time() - t0
        print(c(DIM, f"  [{elapsed:.1f}s]\n"))
        history.append({"q": user_input, "a": answer})

# ─── CLI entry point ────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".rst", ".html", ".htm", ".epub"}

def index_folder(folder: str, recursive: bool = False):
    """Scan a folder and index all supported files."""
    if not os.path.isdir(folder):
        print(c(RED, f"  Not a folder: {folder}"))
        return

    # Collect files
    found = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    found.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                found.append(full)

    if not found:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(c(YELLOW, f"  No supported files found in '{folder}'."))
        print(c(DIM,    f"  Supported: {exts}"))
        return

    print(c(CYAN, f"\n  Found {len(found)} file(s) in '{folder}'\n"))
    ok = 0
    for path in sorted(found):
        try:
            add_document(path)
            ok += 1
        except Exception as e:
            print(c(RED, f"  ERROR on {os.path.basename(path)}: {e}"))

    print(c(GREEN, f"\n  Done — {ok}/{len(found)} files indexed."))
    print(c(DIM,   "  Run:  python rag_cli.py chat\n"))


def main():
    parser = argparse.ArgumentParser(
        description="Universal RAG document Q&A CLI (Ollama + Llama 3 + ChromaDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python rag_cli.py folder ./my_papers
              python rag_cli.py folder ./books --recursive
              python rag_cli.py add paper.pdf
              python rag_cli.py add book1.pdf book2.pdf notes.txt
              python rag_cli.py chat
              python rag_cli.py list
              python rag_cli.py clear
        """),
    )
    sub = parser.add_subparsers(dest="cmd")

    p_folder = sub.add_parser("folder", help="Index all documents in a folder")

    p_folder.add_argument("path", metavar="FOLDER", nargs="?",
                      default="/home/seb/GIT/RagMind/pdf",
                      help="Path to folder (default: /home/seb/GIT/RagMind/pdf)")

    p_folder.add_argument("--recursive", "-r", action="store_true",
                          help="Also scan sub-folders")

    p_add = sub.add_parser("add", help="Index one or more individual files")
    p_add.add_argument("files", nargs="+", metavar="FILE")

    sub.add_parser("chat", help="Start interactive Q&A session")
    sub.add_parser("list", help="List indexed documents")
    sub.add_parser("clear", help="Remove all indexed documents")

    args = parser.parse_args()

    if args.cmd == "folder":
        index_folder(args.path, recursive=args.recursive)
    elif args.cmd == "add":
        for f in args.files:
            add_document(f)
    elif args.cmd == "chat":
        chat_repl()
    elif args.cmd == "list":
        list_documents()
    elif args.cmd == "clear":
        clear_collection()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
