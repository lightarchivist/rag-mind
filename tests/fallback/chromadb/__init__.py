"""In-memory stand-in for the parts of chromadb that rag_cli.py uses.

It mirrors the real library's behaviour where the CLI depends on it:
a maximum batch size for add(), a fixed embedding dimension per collection,
ignored duplicate IDs, and an error when deleting a collection that does not exist.
Data persists per path for the life of the test process, like files on disk.
"""

import math

_STORES: dict[str, dict] = {}
MAX_BATCH = 50


class NotFoundError(ValueError):
    pass


def _match(meta, where):
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    for k, v in where.items():
        if isinstance(v, dict) and "$ne" in v:
            if meta.get(k) == v["$ne"]:
                return False
        elif meta.get(k) != v:
            return False
    return True


class Collection:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata or {}
        self._rows: dict[str, dict] = {}
        self._dim = None

    def count(self):
        return len(self._rows)

    def add(self, ids, embeddings, documents=None, metadatas=None):
        if len(ids) > MAX_BATCH:
            raise ValueError(f"Batch size {len(ids)} exceeds maximum batch size {MAX_BATCH}")
        for i, id_ in enumerate(ids):
            emb = embeddings[i]
            if self._dim is None:
                self._dim = len(emb)
            elif len(emb) != self._dim:
                raise ValueError(f"Embedding dimension {len(emb)} does not match collection dimensionality {self._dim}")
            if id_ in self._rows:
                continue                      # real Chroma warns and ignores
            self._rows[id_] = {"embedding": list(emb),
                               "document": documents[i] if documents else None,
                               "metadata": dict(metadatas[i]) if metadatas else {}}

    def get(self, ids=None, where=None, include=None, limit=None):
        out = {"ids": [], "metadatas": [], "documents": []}
        for id_, row in self._rows.items():
            if ids is not None and id_ not in ids:
                continue
            if not _match(row["metadata"], where):
                continue
            out["ids"].append(id_)
            out["metadatas"].append(row["metadata"])
            out["documents"].append(row["document"])
            if limit and len(out["ids"]) >= limit:
                break
        return out

    def delete(self, ids=None, where=None):
        for id_ in list(self._rows):
            if ids is not None and id_ not in ids:
                continue
            if where is not None and not _match(self._rows[id_]["metadata"], where):
                continue
            del self._rows[id_]
        if not self._rows:
            self._dim = None

    def query(self, query_embeddings, n_results=10, include=None, where=None):
        if n_results < 1:
            raise ValueError(f"Number of requested results {n_results}, cannot be negative, or zero.")
        q = query_embeddings[0]
        scored = []
        for id_, row in self._rows.items():
            if not _match(row["metadata"], where):
                continue
            e = row["embedding"]
            dot = sum(a * b for a, b in zip(q, e))
            na = math.sqrt(sum(a * a for a in q)) or 1e-10
            nb = math.sqrt(sum(b * b for b in e)) or 1e-10
            scored.append((1 - dot / (na * nb), id_, row))
        scored.sort(key=lambda t: t[0])
        top = scored[:n_results]
        return {"ids": [[t[1] for t in top]],
                "distances": [[t[0] for t in top]],
                "documents": [[t[2]["document"] for t in top]],
                "metadatas": [[t[2]["metadata"] for t in top]]}


class PersistentClient:
    def __init__(self, path):
        self._cols = _STORES.setdefault(path, {})

    def get_max_batch_size(self):
        return MAX_BATCH

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._cols:
            self._cols[name] = Collection(name, metadata)
        return self._cols[name]

    def get_collection(self, name):
        if name not in self._cols:
            raise NotFoundError(f"Collection {name} does not exist.")
        return self._cols[name]

    def delete_collection(self, name):
        if name not in self._cols:
            raise NotFoundError(f"Collection {name} does not exist.")
        del self._cols[name]

    def list_collections(self):
        return list(self._cols.values())


def _reset():
    _STORES.clear()
