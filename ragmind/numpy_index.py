"""In-memory vector index for the web app, saved as index.npz (vectors) plus
index.json (texts, metadata, how the vectors were made).

Saves are atomic (write to a temp file, then rename), so a crash mid-save can't
corrupt the index. Indexes saved by RagMind 1.x (index.pkl) are read once and
converted; they are then flagged for rebuilding because their vectors were made
without the embedding task prefixes.
"""

import json
import os
import pickle

import numpy as np

from .keyword import KeywordIndex
from .structure import deepest

NPZ, JSON, LEGACY = "index.npz", "index.json", "index.pkl"


def result(text: str, meta: dict, score: float | None = None, key: str | None = None) -> dict:
    """A stored chunk in the shape the pipeline uses."""
    return {"text": text, "source": meta.get("source", "?"), "page": meta.get("page"), "score": score,
            "key": key or meta.get("source"), "chunk_id": meta.get("chunk_id"),
            "heading": meta.get("heading", ""), "title": meta.get("title", "")}


def keyword_texts(documents, metadatas) -> list[str]:
    """What keyword search sees: a chunk's own section heading, then its text."""
    return [f"{deepest(m.get('heading', ''))} {d}" if m.get("heading") else d
            for d, m in zip(documents, metadatas)]


class NumpyIndex:
    def __init__(self, signature: dict | None = None):
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.signature = signature
        self._keywords = None           # built on first keyword search, dropped on change

    def __len__(self):
        return len(self.documents)

    # ── persistence ──
    @classmethod
    def load(cls, folder: str) -> "NumpyIndex":
        jpath, npath = os.path.join(folder, JSON), os.path.join(folder, NPZ)
        if os.path.exists(jpath) and os.path.exists(npath):
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)
            idx = cls(data.get("signature"))
            idx.documents, idx.metadatas = data["documents"], data["metadatas"]
            with np.load(npath) as z:
                idx.vectors = z["vectors"].astype(np.float32)
            return idx
        legacy = os.path.join(folder, LEGACY)
        if os.path.exists(legacy):
            idx = cls.from_legacy_pickle(legacy)
            idx.save(folder)
            os.replace(legacy, legacy + ".migrated")
            return idx
        return cls()

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        jpath, npath = os.path.join(folder, JSON), os.path.join(folder, NPZ)
        with open(npath + ".tmp", "wb") as f:
            np.savez(f, vectors=self.vectors)
        with open(jpath + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"signature": self.signature, "documents": self.documents,
                       "metadatas": self.metadatas}, f)
        os.replace(npath + ".tmp", npath)
        os.replace(jpath + ".tmp", jpath)

    @classmethod
    def from_legacy_pickle(cls, path: str) -> "NumpyIndex":
        with open(path, "rb") as f:
            old = _LegacyUnpickler(f).load()
        idx = cls(signature=None)             # no signature = built by 1.x, needs rebuilding
        if old.embeddings:
            idx.vectors = np.stack([np.asarray(e, dtype=np.float32) for e in old.embeddings])
        idx.documents = list(old.documents)
        idx.metadatas = [dict(m) for m in old.metadatas]
        return idx

    # ── changes ──
    def add(self, vectors, documents, metadatas):
        if not documents:
            return
        new = np.asarray(vectors, dtype=np.float32)
        if len(self) == 0:
            self.vectors = new
        elif new.shape[1] != self.vectors.shape[1]:
            raise ValueError(f"Embedding size {new.shape[1]} doesn't match the index "
                             f"({self.vectors.shape[1]}); rebuild the index.")
        else:
            self.vectors = np.vstack([self.vectors, new])
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._keywords = None

    def delete_source(self, source: str) -> int:
        keep = [i for i, m in enumerate(self.metadatas) if m["source"] != source]
        removed = len(self) - len(keep)
        if removed:
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self.documents = [self.documents[i] for i in keep]
            self.metadatas = [self.metadatas[i] for i in keep]
            self._keywords = None
        return removed

    # ── reading ──
    def sources(self) -> dict[str, int]:
        counts = {}
        for m in self.metadatas:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def titles(self, sources=None) -> dict[str, str]:
        """{source: title} (the file name without extension when no title is stored)."""
        out = {}
        for m in self.metadatas:
            src = m["source"]
            if src not in out and (sources is None or src in sources):
                out[src] = m.get("title") or os.path.splitext(src)[0]
        return out

    def chunks_of(self, source: str) -> list[tuple[str, dict]]:
        rows = [(d, m) for d, m in zip(self.documents, self.metadatas) if m["source"] == source]
        return sorted(rows, key=lambda r: r[1].get("chunk_id", 0))

    def query(self, vector, n: int, source_filter: str = "") -> list[dict]:
        if len(self) == 0 or n < 1:
            return []
        rows = np.arange(len(self))
        if source_filter:
            rows = np.array([i for i, m in enumerate(self.metadatas) if m["source"] == source_filter], dtype=int)
            if rows.size == 0:
                return []
        q = np.asarray(vector, dtype=np.float32)
        matrix = self.vectors[rows]
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) or 1e-10)
        scores = matrix @ q / np.where(norms == 0, 1e-10, norms)
        top = np.argsort(scores)[::-1][:n]
        return [self.row(int(rows[i]), float(scores[i])) for i in top]

    def row(self, i: int, score: float | None = None) -> dict:
        """Stored chunk i as a search result."""
        return result(self.documents[i], self.metadatas[i], score)


    def keyword_search(self, query: str, n: int, source_filter: str = "") -> list[dict]:
        """Passages containing the query's distinctive words (BM25), best first."""
        if len(self) == 0 or n < 1:
            return []
        if self._keywords is None or self._keywords.size != len(self):
            self._keywords = KeywordIndex(keyword_texts(self.documents, self.metadatas))
        allowed = None
        if source_filter:
            allowed = {i for i, m in enumerate(self.metadatas) if m["source"] == source_filter}
        return [self.row(i) for i, _ in self._keywords.search(query, n, allowed)]


class _LegacyVectorIndex:
    """Stand-in for RagMind 1.x's main.VectorIndex, used only while unpickling."""


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickle only what a RagMind 1.x index.pkl contains: the VectorIndex class
    and numpy arrays. Anything else is refused, since pickle can run code."""
    NUMPY = {("numpy", "ndarray"), ("numpy", "dtype"),
             ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
             ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if name == "VectorIndex" and module in ("main", "__main__", "app.main"):
            return _LegacyVectorIndex
        if (module, name) in self.NUMPY:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from index.pkl")
