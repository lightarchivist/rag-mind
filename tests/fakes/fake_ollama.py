"""A small stand-in for the Ollama HTTP API, for offline tests.

Embeddings are a deterministic bag-of-words hash, so passages that share words
with a question really do score higher. Generation echoes a fixed answer (or a
scripted one) as a stream, the way Ollama does.
"""

import hashlib
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 64
WORD = re.compile(r"[a-z0-9]+")
PREFIXES = ("search_document:", "search_query:")


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    for p in PREFIXES:
        if text.startswith(p):
            text = text[len(p):]
    vec = [0.0] * dim
    for word in WORD.findall(text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


DEFAULT_MODELS = [
    {"name": "llama3:latest", "family": "llama", "parameter_size": "8.0B",
     "capabilities": ["completion"], "context_length": 8192, "size": 4_700_000_000},
    {"name": "qwen2.5:7b", "family": "qwen2", "parameter_size": "7.6B",
     "capabilities": ["completion", "tools"], "context_length": 32768, "size": 4_400_000_000},
    {"name": "nomic-embed-text:latest", "family": "nomic-bert", "parameter_size": "137M",
     "capabilities": ["embedding"], "context_length": 2048, "size": 274_000_000},
]


class FakeOllama:
    def __init__(self):
        self.models = [dict(m) for m in DEFAULT_MODELS]
        self.requests: list[tuple[str, dict]] = []
        self.answer = "This is the answer."
        self.answers: list[str] = []        # scripted answers, consumed in order
        self.fail_generate_after = None     # tokens before a mid-stream error
        self.has_embed_endpoint = True      # False = old Ollama (only /api/embeddings)
        self.has_capabilities = True        # False = old Ollama (/api/show lacks it)
        self.embed_dim = DIM
        self.embed_calls_before_failure = None
        self.lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def calls(self, path):
        return [body for p, body in self.requests if p == path]

    def _find(self, name):
        if ":" not in name:
            name += ":latest"
        return next((m for m in self.models if m["name"] == name), None)

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _not_found_page(self):
                body = b"404 page not found"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _model_missing(self, name):
                self._json(404, {"error": f'model "{name}" not found, try pulling it first'})

            def do_GET(self):
                with fake.lock:
                    fake.requests.append((self.path, {}))
                if self.path == "/api/tags":
                    self._json(200, {"models": [
                        {"name": m["name"], "model": m["name"], "size": m["size"],
                         "details": {"family": m["family"], "parameter_size": m["parameter_size"]}}
                        for m in fake.models]})
                else:
                    self._not_found_page()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with fake.lock:
                    fake.requests.append((self.path, body))

                if self.path == "/api/show":
                    m = fake._find(body.get("model") or body.get("name", ""))
                    if not m:
                        return self._model_missing(body.get("model"))
                    out = {"details": {"family": m["family"]},
                           "model_info": {f"{m['family']}.context_length": m["context_length"]}}
                    if fake.has_capabilities:
                        out["capabilities"] = m["capabilities"]
                    return self._json(200, out)

                if self.path in ("/api/embed", "/api/embeddings"):
                    if self.path == "/api/embed" and not fake.has_embed_endpoint:
                        return self._not_found_page()
                    if not fake._find(body["model"]):
                        return self._model_missing(body["model"])
                    if fake.embed_calls_before_failure is not None:
                        if fake.embed_calls_before_failure <= 0:
                            return self._json(500, {"error": "embedding runner crashed"})
                        fake.embed_calls_before_failure -= 1
                    if self.path == "/api/embed":
                        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                        return self._json(200, {"embeddings": [fake_vector(t, fake.embed_dim) for t in inputs]})
                    return self._json(200, {"embedding": fake_vector(body["prompt"], fake.embed_dim)})

                if self.path == "/api/generate":
                    if not fake._find(body["model"]):
                        return self._model_missing(body["model"])
                    text = fake.answers.pop(0) if fake.answers else fake.answer
                    if not body.get("stream", True):
                        return self._json(200, {"response": text, "done": True})
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for i, word in enumerate(text.split(" ")):
                        if fake.fail_generate_after is not None and i >= fake.fail_generate_after:
                            self.wfile.write((json.dumps({"error": "model runner crashed"}) + "\n").encode())
                            return
                        tok = word if i == 0 else " " + word
                        self.wfile.write((json.dumps({"response": tok, "done": False}) + "\n").encode())
                    self.wfile.write((json.dumps({"response": "", "done": True}) + "\n").encode())
                    return

                self._not_found_page()

        return Handler
