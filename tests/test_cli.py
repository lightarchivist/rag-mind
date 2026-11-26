"""rag_cli.py end to end, against the fake Ollama and chromadb (real or stand-in)."""

import builtins
import json

import pytest

from conftest import write


def store(cli):
    return cli.ChromaStore(cli.load_settings(cli.Settings()))


def client(cli):
    s = cli.load_settings(cli.Settings())
    return cli.OllamaClient(s.ollama_base, timeout=10)


def run_chat(cli, monkeypatch, inputs, *args):
    feed = iter(inputs)

    def fake_input(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr(builtins, "input", fake_input)
    return cli.main(["chat", *args])


BOOK = "The ZenithBook Air 15 has a battery life of 14 hours. " * 5 + \
       "The UltraBook Pro 14 weighs 1.2 kg and has two USB-C ports. " * 5


# ── indexing ──────────────────────────────────────────────────────────────────
def test_add_unchanged_changed_and_duplicate(cli, env, capsys):
    path = write(env / "docs" / "notes.txt", BOOK)
    assert cli.main(["add", path]) == 0
    first = store(cli).count()
    assert first > 0
    assert cli.main(["add", path]) == 0
    assert "unchanged" in capsys.readouterr().out
    write(env / "docs" / "notes.txt", BOOK + " A new sentence about printers.")
    assert cli.main(["add", path]) == 0
    out = capsys.readouterr().out
    assert "replaced" in out
    docs = store(cli).documents()
    assert len(docs) == 1                                 # old version gone
    copy = write(env / "other" / "notes.txt", BOOK + " A new sentence about printers.")
    cli.main(["add", copy])
    assert "already indexed as" in capsys.readouterr().out


def test_failed_reindex_keeps_old_version(cli, env, ollama):
    path = write(env / "a.txt", "Version one of the notes. " * 30)
    cli.main(["add", path])
    before = store(cli).documents()
    write(env / "a.txt", "Version two of the notes. " * 300)
    ollama.embed_calls_before_failure = 0
    assert cli.main(["add", path]) == 1
    assert store(cli).documents() == before             # still searchable


def test_interrupted_add_is_redone(cli, env, monkeypatch):
    path = write(env / "a.txt", BOOK * 20)
    s = store(cli)
    calls = {"n": 0}
    real_add = s._add.__func__

    def flaky(self, col, ids, *rest):
        calls["n"] += 1
        # Store only the first half, as if the process died part-way.
        real_add(self, col, ids[:len(ids) // 2], *[r[:len(ids) // 2] for r in rest])
        raise KeyboardInterrupt
    monkeypatch.setattr(cli.ChromaStore, "_add", flaky)
    with pytest.raises(KeyboardInterrupt):
        s.add_file(path, client(cli))
    monkeypatch.setattr(cli.ChromaStore, "_add", real_add)
    result = store(cli).add_file(path, client(cli))
    assert result["status"] == "added"                 # not "unchanged"
    assert store(cli).count() == result["chunks"]


def test_big_document_is_added_in_batches(cli, env):
    import chromadb
    limit = chromadb.PersistentClient(str(env / "probe")).get_max_batch_size()
    if limit > 2000:
        pytest.skip("real chromadb: batch limit too large to exceed quickly")
    path = write(env / "book.txt", " ".join(f"Sentence number {i} is here." for i in range(limit * 25)))
    assert cli.main(["add", path]) == 0
    assert store(cli).count() > limit


def test_one_bad_file_does_not_stop_the_rest(cli, env, capsys):
    good = write(env / "good.txt", BOOK)
    bad = write(env / "bad.pdf", "not a pdf")
    missing = str(env / "missing.txt")
    assert cli.main(["add", bad, missing, good]) == 1
    out = capsys.readouterr().out
    assert "Could not read bad.pdf" in out and "File not found" in out
    assert [d["source"] for d in store(cli).documents()] == ["good.txt"]


def test_missing_embedding_model_is_explained(cli, env, ollama, capsys):
    ollama.models = [m for m in ollama.models if "embed" not in m["name"]]
    assert cli.main(["add", write(env / "a.txt", BOOK)]) == 1
    out = capsys.readouterr().out
    assert "ollama pull nomic-embed-text" in out and "ollama serve" not in out


def test_ollama_down(cli, env, monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_BASE", "http://127.0.0.1:9")
    assert cli.main(["add", write(env / "a.txt", BOOK)]) == 1
    assert "ollama serve" in capsys.readouterr().out


def test_folder_and_list(cli, env, capsys):
    write(env / "f" / "a.txt", BOOK)
    write(env / "f" / "sub" / "b.md", "Markdown notes about ports. " * 10)
    write(env / "f" / "skip.xyz", "ignored")
    assert cli.main(["folder", str(env / "f"), "-r"]) == 0
    cli.main(["list"])
    out = capsys.readouterr().out
    assert "a.txt" in out and "b.md" in out and "skip.xyz" not in out


def test_same_name_in_two_folders_listed_separately(cli, env):
    cli.main(["add", write(env / "x" / "notes.txt", "First notes. " * 20),
              write(env / "y" / "notes.txt", "Second notes. " * 20)])
    labels = [d["label"] for d in store(cli).documents()]
    assert len(labels) == 2 and all(l.endswith("notes.txt") and "/" in l for l in labels)


def test_add_records_index_signature(cli, env):
    cli.main(["add", write(env / "a.txt", BOOK)])
    meta = json.loads((env / "db" / "ragmind_index.json").read_text())
    assert meta["embed_model"] == "nomic-embed-text:latest" and meta["format"] == 2


# ── clear / reindex ───────────────────────────────────────────────────────────
def test_clear_on_fresh_install_and_confirmation(cli, env, monkeypatch, capsys):
    assert cli.main(["clear", "--yes"]) == 0               # nothing indexed: no crash
    cli.main(["add", write(env / "a.txt", BOOK)])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda p="": "n")
    cli.main(["clear"])
    assert store(cli).count() > 0
    monkeypatch.setattr(builtins, "input", lambda p="": "y")
    cli.main(["clear"])
    assert store(cli).count() == 0


def test_clear_without_tty_needs_yes(cli, env, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main(["clear"]) == 1
    assert "--yes" in capsys.readouterr().out


def legacy_index(cli, env):
    """Build an index the way RagMind 1.x did (no signature file, no prefixes, no path)."""
    import chromadb
    db = chromadb.PersistentClient(path=str(env / "db"))
    col = db.get_or_create_collection("rag_documents", metadata={"hnsw:space": "cosine"})
    col.add(ids=["abc_0", "abc_1"], embeddings=[[0.1] * 64, [0.2] * 64],
            documents=["Old chunk about batteries.", "Old chunk about ports."],
            metadatas=[{"source": "gone.pdf", "chunk_id": 0}, {"source": "gone.pdf", "chunk_id": 1}])


def test_old_index_must_be_rebuilt_then_works(cli, env, capsys, monkeypatch):
    legacy_index(cli, env)
    assert run_chat(cli, monkeypatch, ["hi"]) == 1
    assert "reindex" in capsys.readouterr().out
    assert cli.main(["add", write(env / "a.txt", BOOK)]) == 1      # no mixing old + new vectors
    assert cli.main(["reindex"]) == 0
    out = capsys.readouterr().out
    assert "kept stored text" in out
    s = store(cli)
    assert s.problem() is None and s.count() == 2
    assert cli.main(["add", write(env / "a.txt", BOOK)]) == 0


def test_reindex_after_embedding_model_change(cli, env, ollama, monkeypatch, capsys):
    path = write(env / "a.txt", BOOK)
    cli.main(["add", path])
    ollama.models.append({"name": "mxbai-embed-large:latest", "family": "bert", "parameter_size": "335M",
                          "capabilities": ["embedding"], "context_length": 512, "size": 1})
    ollama.embed_dim = 32                                   # different vector size
    monkeypatch.setenv("EMBED_MODEL", "mxbai-embed-large")
    assert run_chat(cli, monkeypatch, ["hi"]) == 1
    assert "mxbai-embed-large" in capsys.readouterr().out
    assert cli.main(["reindex"]) == 0
    assert "re-read from file" in capsys.readouterr().out
    assert store(cli).problem() is None


def test_failed_rebuild_leaves_old_index(cli, env, ollama):
    legacy_index(cli, env)
    ollama.embed_calls_before_failure = 0
    assert cli.main(["reindex"]) == 1
    s = store(cli)
    assert s.count() == 2 and s.problem() is not None
    names = [c if isinstance(c, str) else c.name for c in s.db.list_collections()]
    assert names == ["rag_documents"]                       # half-built copy removed


# ── chat ──────────────────────────────────────────────────────────────────────
def test_chat_answers_with_sources_and_history(cli, env, ollama, monkeypatch, capsys):
    cli.main(["add", write(env / "laptops.txt", BOOK)])
    capsys.readouterr()
    ollama.answers = ["The battery lasts 14 hours.", "ZenithBook Air 15 battery life",
                      "It weighs about the same."]
    assert run_chat(cli, monkeypatch, ["What is the ZenithBook battery life?",
                                       "and its weight?", "/history"]) == 0
    out = capsys.readouterr().out
    assert "laptops.txt" in out and "The battery lasts 14 hours." in out
    assert "(searched for: ZenithBook Air 15 battery life)" in out
    assert "A1 (llama3)" in out and "A2 (llama3)" in out
    prompt = ollama.calls("/api/generate")[-1]["prompt"]
    assert "=== CONVERSATION SO FAR ===" in prompt and "What is the ZenithBook" in prompt


def test_failed_answer_not_saved(cli, env, ollama, monkeypatch, capsys):
    cli.main(["add", write(env / "laptops.txt", BOOK)])
    ollama.fail_generate_after = 1
    run_chat(cli, monkeypatch, ["battery?", "/history"])
    out = capsys.readouterr().out
    assert "model runner crashed" in out and "not saved" in out and "No history yet" in out


def test_chat_uses_model_context_window(cli, env, ollama, monkeypatch):
    cli.main(["add", write(env / "laptops.txt", BOOK)])
    run_chat(cli, monkeypatch, ["battery?"], "--model", "qwen2.5:7b")
    body = ollama.calls("/api/generate")[-1]
    assert body["model"] == "qwen2.5:7b" and body["options"]["num_ctx"] == 8192


# ── model switching ───────────────────────────────────────────────────────────
def test_models_command_lists_chat_models_only(cli, env, capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "llama3" in out and "qwen2.5:7b" in out and "32K ctx" in out
    assert "nomic-embed-text" not in out


def test_models_default_is_saved_and_used(cli, env, ollama, monkeypatch, capsys):
    assert cli.main(["models", "--default", "qwen2.5:7b"]) == 0
    assert json.loads((env / "config.json").read_text())["llm_model"] == "qwen2.5:7b"
    cli.main(["add", write(env / "a.txt", BOOK)])
    run_chat(cli, monkeypatch, ["battery?"])
    assert ollama.calls("/api/generate")[-1]["model"] == "qwen2.5:7b"


def test_unknown_or_embedding_model_rejected(cli, env, capsys):
    assert cli.main(["models", "--default", "mistral"]) == 1
    out = capsys.readouterr().out
    assert "ollama pull mistral" in out and "llama3" in out
    assert cli.main(["models", "--default", "nomic-embed-text"]) == 1
    assert "embedding model" in capsys.readouterr().out


def test_switch_model_mid_chat(cli, env, ollama, monkeypatch, capsys):
    cli.main(["add", write(env / "a.txt", BOOK)])
    monkeypatch.setenv("RAGMIND_REWRITE", "0")
    run_chat(cli, monkeypatch, ["first?", "/model qwen2.5:7b", "second?", "/model nope", "/history"])
    out = capsys.readouterr().out
    assert [b["model"] for b in ollama.calls("/api/generate")] == ["llama3:latest", "qwen2.5:7b"]
    assert "A1 (llama3)" in out and "A2 (qwen2.5:7b)" in out
    assert "ollama pull nope" in out
    assert json.loads((env / "config.json").read_text())["llm_model"] == "qwen2.5:7b"


def test_env_override_is_mentioned(cli, env, monkeypatch, capsys):
    monkeypatch.setenv("LLM_MODEL", "llama3")
    cli.main(["models", "--default", "qwen2.5:7b"])
    assert "LLM_MODEL environment variable is set" in capsys.readouterr().out


# ── broad questions ───────────────────────────────────────────────────────────
PAPER_A = ("Abstract. We propose a hierarchical multi-agent framework for cyber-physical security. " * 3 +
           "The round-table protocol resolves conflicts between subsystem supervisors. " * 8 +
           "Author contributions: Conceptualization, Data curation, Funding acquisition, Formal analysis. " * 3)
PAPER_B = ("Abstract. Software-defined networking with AI improves industrial IoT security. " * 3 +
           "Controllers detect anomalies in flow tables and reroute traffic. " * 8)


def test_broad_question_covers_both_papers(cli, env, ollama, monkeypatch, capsys):
    cli.main(["add", write(env / "a.txt", PAPER_A), write(env / "b.txt", PAPER_B)])
    capsys.readouterr()
    run_chat(cli, monkeypatch, ["summarise / compare the documents"])
    out = capsys.readouterr().out
    assert "passages from all 2" in out
    prompt = ollama.calls("/api/generate")[-1]["prompt"]
    assert "source: a.txt" in prompt and "source: b.txt" in prompt
    assert "Abstract. We propose" in prompt and "Abstract. Software-defined" in prompt
    assert "Funding acquisition" not in prompt
    assert "The passages come from 2 documents" in prompt
    assert "(source: a.txt, abstract)" in prompt and "(source: b.txt, abstract)" in prompt


def test_specific_question_uses_normal_search(cli, env, ollama, monkeypatch, capsys):
    cli.main(["add", write(env / "a.txt", PAPER_A), write(env / "b.txt", PAPER_B)])
    run_chat(cli, monkeypatch, ["How do controllers detect anomalies in flow tables?"])
    assert "passages from all" not in capsys.readouterr().out
    assert "The passages come from" not in ollama.calls("/api/generate")[-1]["prompt"]


def test_reasoning_model_thinking_is_hidden_in_cli(cli, env, ollama, monkeypatch, capsys):
    ollama.models.append({"name": "deepseek-r1:8b", "family": "qwen2", "parameter_size": "8B",
                          "capabilities": ["completion"], "context_length": 32768, "size": 1})
    cli.main(["add", write(env / "laptops.txt", BOOK)])
    ollama.answer = "<think> Let me check the passages. </think> The battery lasts 14 hours."
    run_chat(cli, monkeypatch, ["battery?", "/history"], "--model", "deepseek-r1:8b")
    out = capsys.readouterr().out
    assert "(thinking…)" in out and "The battery lasts 14 hours." in out
    assert "Let me check the passages" not in out


def test_keyword_search_in_cli_store(cli, env):
    write(env / "docs" / "mpls.txt", "General routing notes. " * 40 +
          "Labels 0 to 15 are reserved; label 3 is implicit null for penultimate hop popping. " +
          "More general routing notes. " * 40)
    write(env / "docs" / "other.txt", "Unrelated text about laptops and batteries. " * 60)
    assert cli.main(["folder", str(env / "docs")]) == 0
    s = store(cli)
    hits = s.keyword_search("Which MPLS label values are reserved, and what is label 3 used for?", 3)
    assert hits and "implicit null" in hits[0]["text"] and hits[0]["source"] == "mpls.txt"
    assert hits[0]["key"].endswith("mpls.txt") and hits[0]["chunk_id"] is not None
    assert s.keyword_search("capital of France", 3) == []


def test_add_force_reindexes_an_unchanged_file(cli, env, capsys):
    path = write(env / "docs" / "book.txt", "LDP i− the default label di−tribution protocol. " * 30)
    assert cli.main(["add", path]) == 0
    count = store(cli).count()
    assert cli.main(["add", path]) == 0
    assert "unchanged" in capsys.readouterr().out
    assert cli.main(["add", "--force", path]) == 0
    out = capsys.readouterr().out
    assert "chunks stored" in out and "repaired 60 characters" in out
    assert store(cli).count() == count
    assert "distribution protocol" in store(cli).keyword_search("default label distribution protocol LDP", 1)[0]["text"]


def test_empty_answer_is_an_error_and_not_saved(cli, env, ollama, monkeypatch, capsys):
    cli.main(["add", write(env / "laptops.txt", BOOK)])
    ollama.answer = "<think>long reasoning that used all the room</think>"
    run_chat(cli, monkeypatch, ["battery?", "/history"])
    out = capsys.readouterr().out
    assert "returned no answer" in out and "No history yet" in out
