"""Settings: built-in defaults, overridden by a JSON config file, overridden by
environment variables (highest priority)."""

import json
import os
from dataclasses import dataclass, fields, replace

# Environment variable for each setting.
ENV = {
    "ollama_base": "OLLAMA_BASE",
    "llm_model": "LLM_MODEL",
    "embed_model": "EMBED_MODEL",
    "max_ctx": "RAGMIND_MAX_CTX",
    "top_k": "RAGMIND_TOP_K",
    "min_score": "RAGMIND_MIN_SCORE",
    "rewrite_followups": "RAGMIND_REWRITE",
    "grade_chunks": "RAGMIND_GRADE",
    "keyword_search": "RAGMIND_KEYWORDS",
    "timeout": "RAGMIND_TIMEOUT",
}


@dataclass(frozen=True)
class Settings:
    ollama_base: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embed_model: str = "nomic-embed-text"
    # Upper limit on the context window requested from Ollama. The real limit is
    # min(this, the model's own context length); bigger windows use more memory.
    max_ctx: int = 8192
    answer_reserve: int = 1024        # tokens kept free for the answer itself
    top_k: int = 6                    # passages retrieved per question
    min_score: float = 0.25           # drop passages below this cosine similarity
    chunk_size: int = 500             # characters per chunk
    chunk_overlap: int = 80
    history_turns: int = 4            # most recent Q&A turns offered to the model
    rewrite_followups: bool = True    # rewrite "what about the second one?" before searching
    grade_chunks: bool = False        # ask the model to filter passages (slower)
    keyword_search: bool = True       # combine similarity search with keyword (BM25) search
    timeout: float = 600.0            # seconds to wait for Ollama (first token can be slow)


def default_config_path() -> str:
    return os.environ.get("RAGMIND_CONFIG") or os.path.join(
        os.path.expanduser("~"), ".ragmind", "config.json")


def _coerce(name: str, raw):
    kind = {f.name: f.type for f in fields(Settings)}[name]
    if kind in (bool, "bool"):
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if kind in (int, "int"):
        return int(raw)
    if kind in (float, "float"):
        return float(raw)
    return str(raw)


def read_config_file(path: str | None = None) -> dict:
    path = path or default_config_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise ValueError(f"Could not read settings file {path}: {e}") from e


def load_settings(base: Settings | None = None, config_path: str | None = None) -> Settings:
    settings = base or Settings()
    known = {f.name for f in fields(Settings)}
    updates = {}
    for key, value in read_config_file(config_path).items():
        if key in known:
            updates[key] = _coerce(key, value)
    for key, var in ENV.items():
        if os.environ.get(var, "").strip():
            updates[key] = _coerce(key, os.environ[var])
    settings = replace(settings, **updates)
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return settings


def save_setting(key: str, value, config_path: str | None = None) -> str:
    """Store one setting in the config file (atomically). Returns the file path."""
    if key not in {f.name for f in fields(Settings)}:
        raise KeyError(key)
    path = config_path or default_config_path()
    data = read_config_file(path)
    data[key] = value
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return path
