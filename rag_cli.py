#!/usr/bin/env python3
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
"""
rag_cli.py — Universal RAG-powered CLI for Document Q&A
Scales from a single paper to multiple books using Ollama.

Usage:
    python rag_cli.py add paper.pdf book.epub notes.txt
    python rag_cli.py folder ./my_papers --recursive
    python rag_cli.py chat [--model qwen2.5]
    python rag_cli.py models [--default qwen2.5]
    python rag_cli.py list
    python rag_cli.py reindex
    python rag_cli.py clear [--yes]

Settings (environment variables override ~/.ragmind/config.json):
    OLLAMA_BASE, LLM_MODEL, EMBED_MODEL, RAG_CLI_DB,
    RAGMIND_MAX_CTX, RAGMIND_TOP_K, RAGMIND_MIN_SCORE, RAGMIND_REWRITE, RAGMIND_GRADE

Requirements:
    pip install -r requirements.txt
    ollama pull llama3
    ollama pull nomic-embed-text   # for embeddings
"""

import argparse
import hashlib
import json
import os
import sys
import textwrap
import time

from ragmind import __version__, pipeline, thinking
from ragmind.chunking import chunk_pages
from ragmind.config import ENV, Settings, default_config_path, load_settings, save_setting
from ragmind.extract import SUPPORTED_EXTENSIONS, ExtractionError, extract_pages
from ragmind.ollama import (ModelInfo, ModelNotFound, NotAChatModel, OllamaClient,
                            OllamaError, OllamaUnavailable, short)
from ragmind.keyword import KeywordIndex
from ragmind.prompt import cite, group_sources, tidy_citations

# ─── Colours ──────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[96m"
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; DIM = "\033[2m"


def c(col, text):
    return f"{col}{text}{RESET}" if sys.stdout.isatty() else text


class CLIError(Exception):
    """A problem to report to the user (no traceback)."""


DB_DIR = os.path.expanduser(os.environ.get("RAG_CLI_DB") or "~/.rag_cli_db")
LEGACY_COLLECTION = "rag_documents"
META_FILE = "ragmind_index.json"


# ─── Vector store (ChromaDB) ──────────────────────────────────────────────────
class ChromaStore:
    """ChromaDB collection plus a small JSON file recording which collection is
    active and how its vectors were made (embedding model, index format)."""

    def __init__(self, settings: Settings, path: str = DB_DIR):
        try:
            import chromadb
        except ImportError:
            raise CLIError("Missing chromadb. Run:  pip install chromadb") from None
        self.settings = settings
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.db = chromadb.PersistentClient(path=path)
        self.meta = self._read_meta()
        self._col = None

    # ── metadata file ──
    def _meta_path(self):
        return os.path.join(self.path, META_FILE)

    def _read_meta(self):
        try:
            with open(self._meta_path(), encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, ValueError):
            return None

    def _write_meta(self, collection_name):
        meta = {"collection": collection_name, **pipeline.index_signature(self.settings)}
        tmp = self._meta_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, self._meta_path())
        self.meta = meta

    # ── collection ──
    @property
    def collection_name(self):
        return (self.meta or {}).get("collection", LEGACY_COLLECTION)

    @property
    def col(self):
        if self._col is None:
            self._col = self.db.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"})
        return self._col

    def count(self) -> int:
        return self.col.count()

    def problem(self) -> str | None:
        """Why this index can't be used with the current settings (None if it can)."""
        if self.meta is None:
            return None if self.count() == 0 else pipeline.signature_problem(None, self.settings)
        return pipeline.signature_problem(self.meta, self.settings)

    def require_usable(self):
        why = self.problem()
        if why:
            raise CLIError(f"The index needs rebuilding: {why}.\n"
                           "  Run:  python rag_cli.py reindex")

    def batch_size(self) -> int:
        get = getattr(self.db, "get_max_batch_size", None)
        return max(1, min(get() if get else 5000, 5000))

    def _add(self, col, ids, vectors, documents, metadatas):
        n = self.batch_size()
        for i in range(0, len(ids), n):
            col.add(ids=ids[i:i + n], embeddings=vectors[i:i + n],
                    documents=documents[i:i + n], metadatas=metadatas[i:i + n])

    # ── indexing ──
    def add_file(self, path: str, ollama: OllamaClient, progress=None, force: bool = False) -> dict:
        """Index one file. Returns {"status": added|unchanged|duplicate|empty, ...}.
        New chunks are stored before old ones are removed, so a failure part-way
        never loses the previously indexed version. `force` re-indexes an
        unchanged file (after a RagMind update that reads documents better)."""
        self._keywords = None               # keyword index is rebuilt after changes
        if not os.path.isfile(path):
            raise CLIError(f"File not found: {path}")
        self.require_usable()
        abs_path = os.path.abspath(path)
        with open(abs_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        first = self.col.get(ids=[f"{file_hash}_0"], include=["metadatas"])
        if first["ids"] and not force:
            meta = first["metadatas"][0]
            last = f"{file_hash}_{meta.get('n_chunks', 1) - 1}"
            if self.col.get(ids=[last])["ids"]:
                if meta.get("path") == abs_path:
                    return {"status": "unchanged"}
                return {"status": "duplicate", "of": meta.get("path", meta.get("source"))}
            self.col.delete(where={"file_hash": file_hash})     # earlier run was interrupted

        report = {}
        pages = extract_pages(abs_path, report)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            return {"status": "empty"}
        vectors = pipeline.embed_documents(ollama, self.settings, [ch.text for ch in chunks], progress)
        ids = [f"{file_hash}_{ch.chunk_id}" for ch in chunks]
        metadatas = [self._chunk_meta(abs_path, file_hash, ch, len(chunks)) for ch in chunks]
        if self.meta is None:               # fresh index: record its format before writing to it
            self._write_meta(self.collection_name)
        if force:                           # same content, same ids: the old copy goes first
            self.col.delete(where={"file_hash": file_hash})
        self._add(self.col, ids, vectors, [ch.text for ch in chunks], metadatas)

        # Now that the new version is stored, drop older versions of this file.
        stale = [i for i, m in self._rows(where={"path": abs_path}) if m.get("file_hash") != file_hash]
        stale += [i for i, m in self._rows(where={"source": os.path.basename(abs_path)})
                  if "path" not in m]                          # indexes from RagMind 1.x
        if stale:
            self.col.delete(ids=stale)
        return {"status": "added", "chunks": len(chunks), "replaced": len(stale),
                "pages": sum(1 for p, t in pages if p and t.strip()), **report}

    @staticmethod
    def _chunk_meta(abs_path, file_hash, ch, n_chunks):
        meta = {"source": os.path.basename(abs_path), "path": abs_path, "file_hash": file_hash,
                "chunk_id": ch.chunk_id, "n_chunks": n_chunks}
        if ch.page is not None:                 # Chroma rejects None values
            meta["page"] = ch.page
        return meta

    def _rows(self, where=None, col=None):
        got = (col or self.col).get(where=where, include=["metadatas"])
        return list(zip(got["ids"], got["metadatas"]))

    # ── search ──
    def search(self, vector, n, key=None) -> list[dict]:
        """Closest chunks overall, or within one document (`key` from keys())."""
        where, limit = None, self.count()
        if key is not None:
            doc = self._doc(key)
            if doc is None:
                return []
            where = {"path": doc["path"]} if doc["path"] else {"source": doc["source"]}
            limit = doc["chunks"]
        n = min(n, limit)
        if n < 1:
            return []
        res = self.col.query(query_embeddings=[vector], n_results=n, where=where,
                             include=["documents", "metadatas", "distances"])
        return [{"text": doc, "source": meta.get("source", "?"), "page": meta.get("page"),
                 "score": 1 - dist, "key": meta.get("path") or meta.get("source"),
                 "chunk_id": meta.get("chunk_id")}
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def keyword_search(self, query, n) -> list[dict]:
        """Passages containing the query's distinctive words (BM25), best first.
        The keyword index is built from the collection on first use and rebuilt
        after any change."""
        stamp = (self.collection_name, self.count())
        if getattr(self, "_keywords", None) is None or self._keywords[0] != stamp:
            got = self.col.get(include=["documents", "metadatas"])
            self._keywords = (stamp, KeywordIndex(got["documents"]), got["documents"], got["metadatas"])
        _, index, texts, metas = self._keywords
        return [{"text": texts[i], "source": metas[i].get("source", "?"), "page": metas[i].get("page"),
                 "score": None, "key": metas[i].get("path") or metas[i].get("source"),
                 "chunk_id": metas[i].get("chunk_id")} for i, _ in index.search(query, n)]

    # ── corpus interface for broad questions (see pipeline.prepare) ──
    def keys(self) -> list[str]:
        self._docs = {d["path"] or d["source"]: d for d in self.documents()}
        return list(self._docs)

    def _doc(self, key):
        if not getattr(self, "_docs", None) or key not in self._docs:
            self.keys()
        return self._docs.get(key)

    def chunk_count(self, key) -> int:
        doc = self._doc(key)
        return doc["chunks"] if doc else 0

    def chunk_range(self, key, start, stop) -> list[dict]:
        """Chunks start..stop-1 of a document, in document order."""
        doc = self._doc(key)
        if not doc or not doc.get("file_hash") or stop <= start:
            return []
        got = self.col.get(ids=[f"{doc['file_hash']}_{i}" for i in range(start, stop)],
                           include=["documents", "metadatas"])
        rows = sorted(zip(got["documents"], got["metadatas"]), key=lambda r: r[1].get("chunk_id", 0))
        return [{"text": text, "source": m.get("source", "?"), "page": m.get("page"), "score": None,
                 "chunk_id": m.get("chunk_id")} for text, m in rows]

    # ── listing ──
    def documents(self) -> list[dict]:
        groups = {}
        for _, m in self._rows():
            key = m.get("path") or m.get("source", "?")
            g = groups.setdefault(key, {"source": m.get("source", "?"), "path": m.get("path"),
                                        "file_hash": m.get("file_hash"), "chunks": 0})
            g["chunks"] += 1
        docs = sorted(groups.values(), key=lambda d: (d["source"].lower(), d["path"] or ""))
        names = [d["source"] for d in docs]
        for d in docs:                          # same file name in two folders: show the path
            d["label"] = d["path"] if names.count(d["source"]) > 1 and d["path"] else d["source"]
        return docs

    # ── maintenance ──
    def _collection_names(self):
        return [col if isinstance(col, str) else col.name for col in self.db.list_collections()]

    def clear(self) -> int:
        self._keywords = None               # keyword index is rebuilt after changes
        removed = 0
        for name in self._collection_names():
            if name == LEGACY_COLLECTION or name.startswith(LEGACY_COLLECTION + "_"):
                removed += self.db.get_collection(name).count()
                self.db.delete_collection(name)
        if os.path.exists(self._meta_path()):
            os.remove(self._meta_path())
        self.meta, self._col = None, None
        return removed

    def reindex(self, ollama: OllamaClient, report=print) -> dict:
        """Rebuild every document with the current settings into a new collection,
        then switch to it. If anything fails, the old index is left untouched."""
        self._keywords = None               # keyword index is rebuilt after changes
        old_name = self.collection_name
        for name in self._collection_names():      # leftovers from an interrupted rebuild
            if name.startswith(LEGACY_COLLECTION + "_") and name != old_name:
                self.db.delete_collection(name)

        groups = {}
        got = self.col.get(include=["metadatas", "documents"])
        for doc, m in zip(got["documents"], got["metadatas"]):
            key = m.get("path") or m.get("source", "?")
            groups.setdefault(key, []).append((m.get("chunk_id", 0), doc, m))

        new_name = f"{LEGACY_COLLECTION}_{int(time.time() * 1000)}"
        new_col = self.db.get_or_create_collection(name=new_name, metadata={"hnsw:space": "cosine"})
        stats = {"from_file": 0, "from_index": 0, "chunks": 0}
        try:
            for key, rows in sorted(groups.items()):
                rows.sort(key=lambda r: r[0])
                path = rows[0][2].get("path")
                if path and os.path.isfile(path):
                    with open(path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                    chunks = chunk_pages(extract_pages(path), self.settings.chunk_size,
                                         self.settings.chunk_overlap)
                    texts = [ch.text for ch in chunks]
                    metas = [self._chunk_meta(path, file_hash, ch, len(chunks)) for ch in chunks]
                    stats["from_file"] += 1
                    how = "re-read from file"
                else:
                    # File moved or deleted (or a 1.x index without paths): keep the stored text.
                    texts = [r[1] for r in rows]
                    file_hash = rows[0][2].get("file_hash") or hashlib.sha256(
                        "".join(texts).encode()).hexdigest()[:16]
                    metas = []
                    for i, r in enumerate(rows):
                        m = {k: v for k, v in r[2].items() if v is not None}
                        m.update(chunk_id=i, n_chunks=len(rows), file_hash=file_hash)
                        metas.append(m)
                    stats["from_index"] += 1
                    how = "file not found; kept stored text"
                if not texts:
                    continue
                vectors = pipeline.embed_documents(ollama, self.settings, texts)
                ids = [f"{file_hash}_{i}" for i in range(len(texts))]
                self._add(new_col, ids, vectors, texts, metas)
                stats["chunks"] += len(texts)
                report(f"  {os.path.basename(key)}: {len(texts)} chunks ({how})")
        except BaseException:
            self.db.delete_collection(new_name)
            raise
        self._write_meta(new_name)
        self._col = None
        if old_name != new_name and old_name in self._collection_names():
            self.db.delete_collection(old_name)
        return stats


# ─── Commands ─────────────────────────────────────────────────────────────────
def add_files(store: ChromaStore, ollama: OllamaClient, paths: list[str], force: bool = False) -> int:
    """Index files one by one; a bad file is reported and skipped. Returns failures."""
    failed = 0
    for path in paths:
        name = os.path.basename(path)
        print(c(DIM, f"  {name}: reading…"), end="", flush=True)

        def dots(done, total):
            print(".", end="", flush=True)
        try:
            result = store.add_file(path, ollama, progress=dots, force=force)
        except (OllamaUnavailable, ModelNotFound):
            print()
            raise                                   # no point trying the other files
        except (CLIError, ExtractionError, OllamaError) as e:
            print(c(RED, f" failed\n    {e}"))
            failed += 1
            continue
        status = result["status"]
        if status == "added":
            pages = f", {result['pages']} pages" if result["pages"] else ""
            msg = f" {result['chunks']} chunks stored{pages}"
            if result["replaced"]:
                msg += f" (replaced {result['replaced']} old chunks)"
            print(c(GREEN, msg))
            if result.get("repaired_s"):
                print(c(DIM, f"    repaired {result['repaired_s']} characters damaged by the PDF's font "
                             "(\"s\" read as \"−\")"))
        elif status == "unchanged":
            print(c(YELLOW, " unchanged since last indexed, skipped"))
        elif status == "duplicate":
            print(c(YELLOW, f" same content already indexed as {result['of']}, skipped"))
        else:
            print(c(RED, " empty or unreadable (no text found; scanned PDFs need OCR first)"))
            failed += 1
    return failed


def find_files(folder: str, recursive: bool) -> list[str]:
    if not os.path.isdir(folder):
        raise CLIError(f"Not a folder: {folder}")
    found = []
    if recursive:
        for root, _, files in os.walk(folder):
            found += [os.path.join(root, f) for f in files
                      if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
    else:
        found = [os.path.join(folder, f) for f in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, f))
                 and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
    return sorted(found)


def index_folder(store, ollama, folder, recursive=False) -> int:
    found = find_files(folder, recursive)
    if not found:
        print(c(YELLOW, f"  No supported files found in '{folder}'."))
        print(c(DIM, f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"))
        return 0
    print(c(CYAN, f"\n  Found {len(found)} file(s) in '{folder}'\n"))
    failed = add_files(store, ollama, found)
    print(c(GREEN if not failed else YELLOW,
            f"\n  Done — {len(found) - failed}/{len(found)} files indexed or up to date."))
    print(c(DIM, "  Run:  python rag_cli.py chat\n"))
    return failed


def list_documents(store: ChromaStore):
    docs = store.documents()
    if not docs:
        print(c(DIM, "  No documents indexed yet. Use:  python rag_cli.py add <file>"))
        return
    print(c(CYAN, "\n── Indexed documents ──────────────────────"))
    for d in docs:
        print(f"  {d['label']:40s} {d['chunks']:5d} chunks")
    why = store.problem()
    if why:
        print(c(YELLOW, f"\n  Needs rebuilding ({why}): python rag_cli.py reindex"))
    print(c(CYAN, "────────────────────────────────────────────\n"))


def clear_index(store: ChromaStore, yes: bool) -> int:
    if not yes:
        if not sys.stdin.isatty():
            raise CLIError("Refusing to clear without confirmation. Use:  python rag_cli.py clear --yes")
        answer = input(c(YELLOW, "  Remove ALL indexed documents? Your files are not touched. [y/N] "))
        if answer.strip().lower() not in ("y", "yes"):
            print(c(DIM, "  Cancelled."))
            return 0
    removed = store.clear()
    print(c(YELLOW, f"  Cleared the index ({removed} chunks removed)."))
    return 0


def reindex(store: ChromaStore, ollama: OllamaClient) -> int:
    if store.count() == 0:
        print(c(DIM, "  Nothing to rebuild: the index is empty."))
        return 0
    print(c(CYAN, f"\n  Rebuilding {len(store.documents())} document(s) with "
                  f"{short(pipeline.index_signature(store.settings)['embed_model'])}…\n"))
    stats = store.reindex(ollama)
    print(c(GREEN, f"\n  Done — {stats['chunks']} chunks."))
    if stats["from_index"]:
        print(c(YELLOW, f"  {stats['from_index']} document(s) were not found on disk; their "
                        "stored text was kept, without page numbers."))
    return 0


# ─── Models ───────────────────────────────────────────────────────────────────
def gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size else ""


def print_models(ollama: OllamaClient, current: str):
    models = ollama.chat_models()
    if not models:
        print(c(YELLOW, "  No chat models installed. Try:  ollama pull llama3"))
        return
    print(c(CYAN, "\n── Installed chat models ───────────────────────────────"))
    for m in models:
        mark = c(GREEN, "*") if short(m.name) == short(current) or m.name == current else " "
        ctx = f"{m.context_length // 1024}K ctx" if m.context_length else ""
        print(f"  {mark} {short(m.name):28s} {m.parameter_size:>7s}  {ctx:>8s}  {gb(m.size):>7s}")
    print(c(DIM, "  * = current.  Get more:  ollama pull <name>   (https://ollama.com/library)"))
    print(c(CYAN, "─────────────────────────────────────────────────────────\n"))


def choose_model(ollama: OllamaClient, name: str) -> ModelInfo:
    try:
        return ollama.resolve_chat_model(name)
    except ModelNotFound as e:
        raise CLIError(str(e)) from None
    except NotAChatModel as e:
        raise CLIError(str(e)) from None


def remember_model(name: str) -> str:
    path = save_setting("llm_model", short(name))
    note = f"  Saved as the default model ({path})."
    if os.environ.get(ENV["llm_model"]):
        note += f"\n  Note: the {ENV['llm_model']} environment variable is set and overrides it."
    return note


def models_command(ollama, settings, default: str | None) -> int:
    if default:
        model = choose_model(ollama, default)
        print(c(GREEN, f"  Default model: {short(model.name)}"))
        print(c(DIM, remember_model(model.name)))
        return 0
    print_models(ollama, settings.llm_model)
    return 0


# ─── Chat REPL ────────────────────────────────────────────────────────────────
BANNER = r"""
  ____             __  __ _           _
 |  _ \ __ _  __ _|  \/  (_)_ __   __| |
 | |_) / _` |/ _` | |\/| | | '_ \ / _` |
 |  _ < (_| | (_| | |  | | | | | | (_| |
 |_| \_\__,_|\__, |_|  |_|_|_| |_|\__,_|
             |___/
  Universal Document Q&A  •  RAG + Ollama  •  v{version}
"""

HELP = """
Commands:
  <question>     Ask anything about indexed documents
  /model         Show installed chat models
  /model <name>  Switch to another model (and remember it as the default)
  /sources       Show which documents are indexed
  /history       Show this conversation
  /reset         Forget this conversation (documents stay indexed)
  /quit          Exit
"""


def answer_question(store, ollama, settings, model: ModelInfo, question, history) -> str | None:
    """Retrieve, prompt and stream one answer. Returns the answer, or None on failure."""
    prepared = pipeline.prepare(ollama, settings, model.name, question, history, store.search,
                                corpus=store)
    if prepared.search_query != question:
        print(c(DIM, f"  (searched for: {prepared.search_query})"))
    if prepared.broad:
        docs = len({ch["source"] for ch in prepared.chunks})
        print(c(DIM, f"  (question about the documents as a whole: passages from all {docs})"))
    if prepared.chunks:
        # Numbered like the passages the model saw, so "[2]" in an answer is source 2.
        for numbers, where, section in group_sources(prepared.chunks):
            section = f" ({section})" if section else ""
            print(c(DIM, f"  [{', '.join(map(str, numbers))}] {where}{section}"))
    else:
        print(c(DIM, "  No passages matched closely; the model will say if it can't answer."))

    print(c(CYAN, f"\n{short(model.name)} › "), end="", flush=True)
    hold = thinking.reasons_unseparated(model)
    if hold:
        print(c(DIM, "(thinking…) "), end="", flush=True)
    t0, parts = time.time(), []
    try:
        tokens = ollama.generate_stream(model.name, prepared.prompt,
                                        pipeline.generation_options(prepared.num_ctx))
        for tok in map(tidy_citations, thinking.visible(tokens, hold=hold)):
            if hold and not parts:
                print()                         # answer starts on the line below "(thinking…)"
            print(tok, end="", flush=True)
            parts.append(tok)
    except OllamaError as e:
        print(c(RED, f"\n  ERROR: {e}\n  (This answer was not saved to the conversation.)\n"))
        return None
    except KeyboardInterrupt:
        print(c(DIM, "\n  (stopped; not saved to the conversation)\n"))
        return None
    print(c(DIM, f"\n  [{time.time() - t0:.1f}s]\n"))
    return "".join(parts)


def chat_repl(store: ChromaStore, ollama: OllamaClient, settings: Settings, model_name: str) -> int:
    if store.count() == 0:
        print(c(YELLOW, "No documents indexed. Add some first:\n  python rag_cli.py add paper.pdf"))
        return 1
    store.require_usable()
    model = choose_model(ollama, model_name)
    docs = store.documents()

    print(c(CYAN, BANNER.format(version=__version__)))
    print(c(BOLD, "  Indexed: ") + ", ".join(d["label"] for d in docs))
    print(c(BOLD, "  Chunks : ") + str(store.count()))
    print(c(BOLD, "  Model  : ") + short(model.name))
    print(c(DIM, "  Type /help for commands, /quit to exit\n"))

    history = []
    while True:
        try:
            user_input = input(c(GREEN, "You › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(c(DIM, "\nGoodbye!"))
            return 0
        if not user_input:
            continue
        cmd, _, arg = user_input.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()

        if cmd in ("/quit", "/exit", "/q"):
            print(c(DIM, "Goodbye!"))
            return 0
        if cmd == "/help":
            print(c(CYAN, HELP))
        elif cmd in ("/sources", "/list"):
            list_documents(store)
        elif cmd in ("/model", "/models"):
            if not arg:
                print_models(ollama, model.name)
                continue
            try:
                model = choose_model(ollama, arg)
            except CLIError as e:
                print(c(RED, f"  {e}\n"))
                continue
            print(c(GREEN, f"  Now using {short(model.name)}."))
            print(c(DIM, remember_model(model.name) + "\n"))
        elif cmd == "/history":
            if not history:
                print(c(DIM, "  No history yet.\n"))
            for i, h in enumerate(history, 1):
                print(c(BOLD, f"\n  Q{i}: ") + h["q"])
                print(c(DIM, f"  A{i} ({h['model']}): ") + h["a"][:200] + ("…" if len(h["a"]) > 200 else ""))
            print()
        elif cmd in ("/reset", "/clear"):
            history.clear()
            print(c(DIM, "  Conversation forgotten (documents stay indexed).\n"))
        elif cmd.startswith("/"):
            print(c(YELLOW, f"  Unknown command {cmd}. Type /help.\n"))
        else:
            try:
                answer = answer_question(store, ollama, settings, model, user_input, history)
            except OllamaError as e:
                print(c(RED, f"  ERROR: {e}\n"))
                continue
            if answer is not None:
                history.append({"q": user_input, "a": answer, "model": short(model.name)})


# ─── CLI entry point ──────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Universal RAG document Q&A CLI (Ollama + ChromaDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python rag_cli.py folder ./my_papers
              python rag_cli.py folder ./books --recursive
              python rag_cli.py add paper.pdf book1.pdf notes.txt
              python rag_cli.py chat
              python rag_cli.py chat --model qwen2.5
              python rag_cli.py models
              python rag_cli.py models --default qwen2.5
              python rag_cli.py list
              python rag_cli.py reindex
              python rag_cli.py clear
        """),
    )
    parser.add_argument("--version", action="version", version=f"RagMind {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_folder = sub.add_parser("folder", help="Index all documents in a folder")
    p_folder.add_argument("path", metavar="FOLDER", nargs="?",
                          default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf"),
                          help="Path to folder (default: the pdf/ folder next to this script)")
    p_folder.add_argument("--recursive", "-r", action="store_true", help="Also scan sub-folders")

    p_add = sub.add_parser("add", help="Index one or more individual files")
    p_add.add_argument("files", nargs="+", metavar="FILE")
    p_add.add_argument("--force", action="store_true",
                       help="Re-index files even if unchanged (after a RagMind update)")

    p_chat = sub.add_parser("chat", help="Start interactive Q&A session")
    p_chat.add_argument("--model", "-m", help="Chat model for this session (see: models)")

    p_models = sub.add_parser("models", help="List installed chat models or set the default")
    p_models.add_argument("--default", metavar="NAME", help="Remember NAME as the default chat model")

    sub.add_parser("list", help="List indexed documents")
    sub.add_parser("reindex", help="Rebuild the index (after an upgrade or embedding-model change)")
    p_clear = sub.add_parser("clear", help="Remove all indexed documents")
    p_clear.add_argument("--yes", "-y", action="store_true", help="Don't ask for confirmation")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 0
    try:
        settings = load_settings(Settings())
        ollama = OllamaClient(settings.ollama_base, timeout=settings.timeout)
        if args.cmd == "models":
            return models_command(ollama, settings, args.default)
        store = ChromaStore(settings)
        if args.cmd == "folder":
            return 1 if index_folder(store, ollama, args.path, args.recursive) else 0
        if args.cmd == "add":
            return 1 if add_files(store, ollama, args.files, force=args.force) else 0
        if args.cmd == "chat":
            return chat_repl(store, ollama, settings, args.model or settings.llm_model)
        if args.cmd == "list":
            list_documents(store)
            return 0
        if args.cmd == "reindex":
            return reindex(store, ollama)
        if args.cmd == "clear":
            return clear_index(store, args.yes)
    except (CLIError, OllamaError, ExtractionError, ValueError) as e:
        print(c(RED, f"ERROR: {e}"))
        return 1
    except KeyboardInterrupt:
        print(c(DIM, "\nInterrupted."))
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
