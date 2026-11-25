"""Shared test setup.

- A fake Ollama server (tests/fakes/fake_ollama.py) runs for each test that asks for it.
- FastAPI is always replaced by a tiny stub (tests/stubs) so endpoint functions
  can be called directly.
- chromadb: the real library is used when installed; otherwise an in-memory
  stand-in (tests/fallback) is used.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS / "stubs"))
if importlib.util.find_spec("chromadb") is None:
    sys.path.append(str(TESTS / "fallback"))
sys.path.insert(0, str(TESTS / "fakes"))

from fake_ollama import FakeOllama  # noqa: E402

# Never let a test talk to a real Ollama or read the user's settings.
for var in ("OLLAMA_BASE", "LLM_MODEL", "EMBED_MODEL", "RAG_CLI_DB", "RAGMIND_CONFIG",
            "RAGMIND_MAX_CTX", "RAGMIND_TOP_K", "RAGMIND_MIN_SCORE", "RAGMIND_GRADE",
            "RAGMIND_REWRITE", "DATA_DIR"):
    os.environ.pop(var, None)


@pytest.fixture
def ollama():
    fake = FakeOllama().start()
    yield fake
    fake.stop()


@pytest.fixture
def env(ollama, tmp_path, monkeypatch):
    """Point every setting at the fake server and a temp folder."""
    monkeypatch.setenv("OLLAMA_BASE", ollama.url)
    monkeypatch.setenv("RAG_CLI_DB", str(tmp_path / "db"))
    monkeypatch.setenv("RAGMIND_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return tmp_path


@pytest.fixture
def cli(env):
    """A fresh import of rag_cli.py with settings read from the env fixture."""
    sys.modules.pop("rag_cli", None)
    import rag_cli
    return importlib.reload(rag_cli)


@pytest.fixture
def web(env):
    """A fresh import of web/app/main.py with settings read from the env fixture."""
    sys.path.insert(0, str(ROOT / "web" / "app"))
    try:
        sys.modules.pop("main", None)
        import main
        yield importlib.reload(main)
    finally:
        sys.path.remove(str(ROOT / "web" / "app"))
        sys.modules.pop("main", None)


def write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)
