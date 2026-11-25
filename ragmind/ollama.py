"""Small Ollama HTTP client (standard library only) with clear errors."""

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator


class OllamaError(Exception):
    """Any failure talking to Ollama. The message is meant for the user."""


class OllamaUnavailable(OllamaError):
    pass


class OllamaTimeout(OllamaError):
    pass


class ModelNotFound(OllamaError):
    def __init__(self, model: str, installed: list[str] | None = None):
        self.model = model
        self.installed = installed or []
        msg = f"Model '{model}' is not installed in Ollama."
        if self.installed:
            msg += " Installed: " + ", ".join(self.installed) + "."
        msg += f"\n  Install it with:  ollama pull {model}"
        super().__init__(msg)


class NotAChatModel(OllamaError):
    def __init__(self, model: str):
        self.model = model
        super().__init__(f"'{model}' is an embedding model; it can't answer questions. "
                         "Pick a chat model (see the models list).")


def normalize(name: str) -> str:
    """Ollama treats 'llama3' and 'llama3:latest' as the same model."""
    name = (name or "").strip()
    return name if ":" in name else f"{name}:latest"


def short(name: str) -> str:
    return name[:-len(":latest")] if name.endswith(":latest") else name


@dataclass
class ModelInfo:
    name: str
    size: int = 0
    family: str = ""
    parameter_size: str = ""
    capabilities: list[str] | None = None
    context_length: int | None = None

    @property
    def is_embedding(self) -> bool:
        if self.capabilities is not None:
            return "embedding" in self.capabilities and "completion" not in self.capabilities
        # Older Ollama: no capabilities field, so guess from family and name.
        return self.family in ("bert", "nomic-bert") or "embed" in self.name.lower()

    def to_dict(self) -> dict:
        return {"name": short(self.name), "size": self.size, "family": self.family,
                "parameter_size": self.parameter_size, "context_length": self.context_length}


@dataclass
class OllamaClient:
    base: str
    timeout: float = 600.0
    _embed_endpoint: bool | None = field(default=None, repr=False)
    _show_cache: dict = field(default_factory=dict, repr=False)

    # ── transport ─────────────────────────────────────────────────────────────
    def _open(self, path: str, payload: dict | None = None, timeout: float | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self.base.rstrip('/')}{path}", data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET")
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            raise self._http_error(e, payload) from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise OllamaTimeout(self._timeout_msg()) from None
            raise OllamaUnavailable(
                f"Cannot reach Ollama at {self.base} ({e.reason}).\n"
                "  Start it with:  ollama serve") from None
        except (TimeoutError, socket.timeout):
            raise OllamaTimeout(self._timeout_msg()) from None
        except OSError as e:
            raise OllamaUnavailable(f"Lost connection to Ollama at {self.base}: {e}") from None

    def _timeout_msg(self):
        return (f"Ollama did not respond within {self.timeout:.0f}s. A large model may still be "
                "loading; try again, or raise RAGMIND_TIMEOUT.")

    def _http_error(self, e: urllib.error.HTTPError, payload) -> OllamaError:
        try:
            body = e.read().decode("utf-8", "replace")
        except OSError:
            body = ""
        try:
            message = json.loads(body).get("error", body)
        except ValueError:
            message = body
        if e.code == 404 and "not found" in message and "model" in message:
            model = (payload or {}).get("model", "?")
            err = ModelNotFound(short(normalize(model)))
            err.http_status = 404
            return err
        err = OllamaError(f"Ollama returned HTTP {e.code}: {message.strip() or e.reason}")
        err.http_status = e.code
        err.plain_404 = e.code == 404 and not message.lstrip().startswith("{") and "model" not in message
        return err

    def _json(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        try:
            with self._open(path, payload, timeout) as r:
                return json.loads(r.read())
        except (TimeoutError, socket.timeout):
            raise OllamaTimeout(self._timeout_msg()) from None
        except ValueError as e:
            raise OllamaError(f"Ollama sent an unreadable reply: {e}") from None

    # ── models ────────────────────────────────────────────────────────────────
    def ping(self, timeout: float = 5) -> bool:
        try:
            self._json("/api/tags", timeout=timeout)
            return True
        except OllamaError:
            return False

    def list_models(self) -> list[ModelInfo]:
        data = self._json("/api/tags", timeout=30)
        out = []
        for m in data.get("models", []):
            d = m.get("details") or {}
            out.append(ModelInfo(name=m.get("name") or m.get("model", ""), size=m.get("size", 0),
                                 family=d.get("family", ""), parameter_size=d.get("parameter_size", "")))
        return out

    def show(self, model: str) -> dict:
        key = normalize(model)
        if key not in self._show_cache:
            self._show_cache[key] = self._json("/api/show", {"model": key}, timeout=30)
        return self._show_cache[key]

    def model_details(self, info: ModelInfo) -> ModelInfo:
        """Fill in capabilities and context length from /api/show (best effort)."""
        try:
            data = self.show(info.name)
        except OllamaError:
            return info
        info.capabilities = data.get("capabilities")
        info.context_length = _context_length(data)
        info.family = info.family or (data.get("details") or {}).get("family", "")
        return info

    def chat_models(self) -> list[ModelInfo]:
        models = [self.model_details(m) for m in self.list_models()]
        return sorted((m for m in models if not m.is_embedding), key=lambda m: m.name)

    def resolve_chat_model(self, name: str) -> ModelInfo:
        """Return the installed chat model called `name`, or raise a helpful error."""
        wanted = normalize(name)
        installed = self.list_models()
        match = next((m for m in installed if normalize(m.name) == wanted), None)
        if match is None:
            raise ModelNotFound(short(wanted), [short(m.name) for m in installed if not
                                                self.model_details(m).is_embedding])
        self.model_details(match)
        if match.is_embedding:
            raise NotAChatModel(short(wanted))
        return match

    def context_length(self, model: str) -> int | None:
        try:
            return _context_length(self.show(model))
        except OllamaError:
            return None

    # ── embeddings ────────────────────────────────────────────────────────────
    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embed_endpoint is not False:
            try:
                data = self._json("/api/embed", {"model": model, "input": texts})
                self._embed_endpoint = True
                vectors = data.get("embeddings") or []
                if len(vectors) != len(texts):
                    raise OllamaError("Ollama returned the wrong number of embeddings.")
                return vectors
            except OllamaError as e:
                if not getattr(e, "plain_404", False):
                    raise
                self._embed_endpoint = False      # Ollama older than 0.3: one text per call
        return [self._json("/api/embeddings", {"model": model, "prompt": t})["embedding"]
                for t in texts]

    # ── generation ────────────────────────────────────────────────────────────
    def generate(self, model: str, prompt: str, options: dict | None = None) -> str:
        data = self._json("/api/generate", {"model": model, "prompt": prompt,
                                            "stream": False, "options": options or {}})
        if "error" in data:
            raise OllamaError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    def generate_stream(self, model: str, prompt: str, options: dict | None = None) -> Iterator[str]:
        """Yield answer tokens. Raises OllamaError if the answer stops early."""
        resp = self._open("/api/generate", {"model": model, "prompt": prompt,
                                            "stream": True, "options": options or {}})
        done = False
        try:
            with resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(f"Ollama stopped with an error: {chunk['error']}")
                    tok = chunk.get("response", "")
                    if tok:
                        yield tok
                    if chunk.get("done"):
                        done = True
                        break
        except (TimeoutError, socket.timeout):
            raise OllamaTimeout(self._timeout_msg()) from None
        except ValueError as e:
            raise OllamaError(f"Ollama sent an unreadable reply: {e}") from None
        except OSError as e:
            raise OllamaUnavailable(f"Lost connection to Ollama: {e}") from None
        if not done:
            raise OllamaError("The answer was cut off: Ollama closed the connection early.")


def _context_length(show_data: dict) -> int | None:
    for key, value in (show_data.get("model_info") or {}).items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None
