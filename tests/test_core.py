"""Shared core: chunking, extraction, prompt budget, settings, Ollama client, pipeline."""

import json
import pickle
import sys
import types

import numpy as np
import pytest

from conftest import write
from ragmind import pipeline
from ragmind.chunking import chunk_pages, chunk_text
from ragmind.config import Settings, load_settings, save_setting
from ragmind.extract import ExtractionError, extract_pages
from ragmind.numpy_index import NumpyIndex
from ragmind.ollama import (ModelNotFound, NotAChatModel, OllamaClient, OllamaError,
                            OllamaUnavailable, normalize)
from ragmind.prompt import SYSTEM, build_prompt, estimate_tokens


# ── chunking ──────────────────────────────────────────────────────────────────
def test_chunks_cover_text_with_overlap_and_end():
    text = " ".join(f"Sentence {i} has some words in it." for i in range(400))
    chunks = chunk_text(text, 500, 80)
    assert chunks[0].text.startswith("Sentence 0 ")
    assert chunks[-1].text.endswith("Sentence 399 has some words in it.")
    assert all(len(c.text) <= 500 for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert all(c.text.endswith(".") for c in chunks[:-1])       # sentence boundaries


@pytest.mark.parametrize("text", ["", "   ", "x", "a" * 5000, "no punctuation " * 300])
def test_chunking_terminates_on_awkward_input(text):
    chunks = chunk_text(text, 500, 80)
    assert len(chunks) <= len(text) + 1


def test_chunks_record_starting_page():
    pages = [(1, "Alpha. " * 100), (2, ""), (3, "Gamma. " * 100)]
    chunks = chunk_pages(pages, 300, 40)
    assert chunks[0].page == 1
    assert chunks[-1].page == 3
    assert {c.page for c in chunks} == {1, 3}


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", 100, 100)


# ── extraction ────────────────────────────────────────────────────────────────
def test_unsupported_type_raises_instead_of_exiting(tmp_path):
    with pytest.raises(ExtractionError, match="Unsupported"):
        extract_pages(write(tmp_path / "a.xyz", "hi"))


def test_corrupt_pdf_gives_readable_error(tmp_path):
    with pytest.raises(ExtractionError, match="Could not read bad.pdf"):
        extract_pages(write(tmp_path / "bad.pdf", "not a pdf"))


def test_html_drops_scripts(tmp_path):
    html = "<html><script>var secret=1</script><style>p{}</style><p>Visible text</p></html>"
    text = extract_pages(write(tmp_path / "a.html", html))[0][1]
    assert "Visible text" in text and "secret" not in text


def test_docx_includes_tables(tmp_path):
    import docx
    d = docx.Document()
    d.add_paragraph("Before the table.")
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Price", "42 EUR"
    d.add_paragraph("After the table.")
    d.save(tmp_path / "a.docx")
    text = extract_pages(str(tmp_path / "a.docx"))[0][1]
    assert text.index("Before") < text.index("Price | 42 EUR") < text.index("After")


def test_pdf_pages_are_numbered(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    w.add_blank_page(100, 100)
    w.add_blank_page(100, 100)
    w.write(str(tmp_path / "blank.pdf"))
    assert [p for p, _ in extract_pages(str(tmp_path / "blank.pdf"))] == [1, 2]


# ── prompt budget ─────────────────────────────────────────────────────────────
def chunk(i, n=500, score=0.9):
    return {"text": f"passage{i} " + "x" * n, "source": "a.pdf", "page": i, "score": score}


def test_prompt_fits_budget_and_keeps_rules_and_question():
    history = [{"q": f"q{i}", "a": "word " * 2000} for i in range(4)]
    prompt, used, turns = build_prompt("Final question?", [chunk(i) for i in range(6)],
                                       history, ctx_tokens=4096, answer_reserve=1024)
    assert estimate_tokens(prompt) <= 4096 - 1024
    assert prompt.startswith(SYSTEM)
    assert prompt.rstrip().endswith("User: Final question?\nAssistant:")
    assert used and turns >= 1                      # latest turn kept for follow-ups
    assert "q3" in prompt                           # the newest turn, not the oldest


def test_prompt_drops_lowest_passages_first_when_tight():
    prompt, used, _ = build_prompt("Q?", [chunk(i, 3000) for i in range(6)], [], 4096, 1024)
    assert [c["page"] for c in used] == list(range(len(used)))
    assert 0 < len(used) < 6


def test_prompt_without_passages_says_so():
    prompt, used, _ = build_prompt("Q?", [], [], 8192)
    assert "No passages matched" in prompt and used == []


def test_prompt_cites_pages():
    prompt, _, _ = build_prompt("Q?", [chunk(7)], [], 8192)
    assert "source: a.pdf, p. 7" in prompt


# ── settings ──────────────────────────────────────────────────────────────────
def test_settings_precedence(tmp_path, monkeypatch):
    cfg = str(tmp_path / "c.json")
    save_setting("llm_model", "qwen2.5", cfg)
    save_setting("top_k", 3, cfg)
    assert load_settings(config_path=cfg).llm_model == "qwen2.5"
    monkeypatch.setenv("LLM_MODEL", "mistral")
    monkeypatch.setenv("RAGMIND_GRADE", "yes")
    s = load_settings(config_path=cfg)
    assert (s.llm_model, s.top_k, s.grade_chunks) == ("mistral", 3, True)


def test_bad_config_file_is_reported(tmp_path):
    (tmp_path / "c.json").write_text("{not json")
    with pytest.raises(ValueError, match="Could not read settings"):
        load_settings(config_path=str(tmp_path / "c.json"))


# ── Ollama client ─────────────────────────────────────────────────────────────
def test_unreachable_ollama():
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        OllamaClient("http://127.0.0.1:9").list_models()


def test_missing_model_is_not_reported_as_ollama_down(ollama):
    client = OllamaClient(ollama.url)
    with pytest.raises(ModelNotFound, match="ollama pull mistral"):
        client.embed("mistral", ["hi"])


def test_batched_embed_and_old_ollama_fallback(ollama):
    client = OllamaClient(ollama.url)
    assert len(client.embed("nomic-embed-text", ["a", "b", "c"])) == 3
    assert len(ollama.calls("/api/embed")) == 1
    ollama.has_embed_endpoint = False
    old = OllamaClient(ollama.url)
    assert len(old.embed("nomic-embed-text", ["a", "b"])) == 2
    assert len(ollama.calls("/api/embeddings")) == 2


def test_stream_error_midway_raises(ollama):
    ollama.fail_generate_after = 2
    client = OllamaClient(ollama.url)
    got = []
    with pytest.raises(OllamaError, match="model runner crashed"):
        for tok in client.generate_stream("llama3", "hi"):
            got.append(tok)
    assert "".join(got) == "This is"


def test_chat_models_exclude_embedding_models(ollama):
    names = [m.name for m in OllamaClient(ollama.url).chat_models()]
    assert names == ["llama3:latest", "qwen2.5:7b"]
    ollama.has_capabilities = False                    # older Ollama: guess by family
    names = [m.name for m in OllamaClient(ollama.url).chat_models()]
    assert "nomic-embed-text:latest" not in names


def test_resolve_chat_model(ollama):
    client = OllamaClient(ollama.url)
    assert client.resolve_chat_model("llama3").name == "llama3:latest"
    assert client.resolve_chat_model("qwen2.5:7b").context_length == 32768
    with pytest.raises(ModelNotFound) as e:
        client.resolve_chat_model("mistral")
    assert e.value.installed == ["llama3", "qwen2.5:7b"]
    with pytest.raises(NotAChatModel):
        client.resolve_chat_model("nomic-embed-text")


def test_normalize():
    assert normalize("llama3") == "llama3:latest"
    assert normalize("qwen2.5:7b") == "qwen2.5:7b"


# ── pipeline ──────────────────────────────────────────────────────────────────
def test_nomic_prefixes_are_used(ollama):
    client, s = OllamaClient(ollama.url), Settings()
    pipeline.embed_documents(client, s, ["doc text"])
    pipeline.embed_query(client, s, "question")
    sent = [b["input"][0] for b in ollama.calls("/api/embed")]
    assert sent == ["search_document: doc text", "search_query: question"]


def test_context_window_follows_model_capped_by_setting(ollama):
    client = OllamaClient(ollama.url)
    assert pipeline.context_window(client, Settings(max_ctx=8192), "qwen2.5:7b") == 8192
    assert pipeline.context_window(client, Settings(max_ctx=65536), "qwen2.5:7b") == 32768
    assert pipeline.context_window(client, Settings(max_ctx=65536), "llama3") == 8192


def test_followup_is_rewritten_for_search_only(ollama):
    client = OllamaClient(ollama.url)
    ollama.answers = ["What is the battery life of the ZenithBook Air 15?"]
    seen = []
    prepared = pipeline.prepare(client, Settings(), "llama3", "and its battery?",
                                [{"q": "Tell me about the ZenithBook Air 15", "a": "It is a laptop."}],
                                lambda v, n: seen.append(v) or [])
    assert prepared.search_query.startswith("What is the battery life")
    assert "User: and its battery?" in prepared.prompt


@pytest.mark.parametrize("reply", [
    "What is the weight of the UltraBook Pro 14?",
    "Here is the rewritten question:\n\n\"What is the weight of the UltraBook Pro 14?\"",
    "Rewritten question: What is the weight of the UltraBook Pro 14?",
    "**What is the weight of the UltraBook Pro 14?**",
])
def test_rewrite_handles_chatty_models(ollama, reply):
    ollama.answers = [reply]
    q = pipeline.rewrite_followup(OllamaClient(ollama.url), "llama3", "and its weight?",
                                  [{"q": "Tell me about the UltraBook Pro 14", "a": "A laptop."}], 4096)
    assert q == "What is the weight of the UltraBook Pro 14?"


def test_grading_accepts_formatted_yes(ollama):
    ollama.answers = ["**Yes**", "No."]
    chunks = [{"text": "a", "score": 1}, {"text": "b", "score": 1}]
    kept = pipeline.grade_chunks(OllamaClient(ollama.url), "llama3", "q", chunks, 4096)
    assert [c["text"] for c in kept] == ["a"]


def test_rewrite_falls_back_on_rambling_reply(ollama):
    ollama.answers = ["x " * 400]
    q = pipeline.rewrite_followup(OllamaClient(ollama.url), "llama3", "and then?",
                                  [{"q": "a", "a": "b"}], 4096)
    assert q == "and then?"


def test_min_score_and_grading(ollama):
    results = [{"text": "t1", "source": "a", "page": None, "score": 0.9},
               {"text": "t2", "source": "a", "page": None, "score": 0.1}]
    assert [r["text"] for r in pipeline.select(results, 6, 0.25)] == ["t1"]
    client = OllamaClient(ollama.url)
    ollama.answers = ["no", "yes"]
    two = [dict(results[0]), dict(results[0], text="t3")]
    assert [r["text"] for r in pipeline.grade_chunks(client, "llama3", "q", two, 4096)] == ["t3"]
    ollama.answers = ["no", "no"]                     # all rejected: keep them rather than nothing
    assert len(pipeline.grade_chunks(client, "llama3", "q", two, 4096)) == 2


def test_signature_problems():
    s = Settings()
    assert pipeline.signature_problem(pipeline.index_signature(s), s) is None
    assert "older version" in pipeline.signature_problem(None, s)
    other = dict(pipeline.index_signature(s), embed_model="mxbai-embed-large:latest")
    assert "mxbai" in pipeline.signature_problem(other, s)


# ── numpy index (web) ─────────────────────────────────────────────────────────
def test_numpy_index_roundtrip_and_query(tmp_path):
    idx = NumpyIndex({"embed_model": "x", "format": 2})
    idx.add([[1, 0], [0, 1]], ["one", "two"], [{"source": "a", "chunk_id": 0, "page": 3},
                                              {"source": "b", "chunk_id": 0}])
    idx.save(str(tmp_path))
    again = NumpyIndex.load(str(tmp_path))
    top = again.query([1, 0.1], 1)[0]
    assert (top["text"], top["page"]) == ("one", 3)
    assert again.query([1, 0], 5, source_filter="b")[0]["text"] == "two"
    assert again.delete_source("a") == 1 and again.sources() == {"b": 1}
    with pytest.raises(ValueError, match="rebuild"):
        again.add([[1, 2, 3]], ["bad"], [{"source": "c"}])


def test_legacy_pickle_is_migrated_safely(tmp_path):
    # Build a pickle exactly like RagMind 1.x did: main.VectorIndex with numpy arrays.
    main = types.ModuleType("main")
    exec("class VectorIndex:\n    pass", main.__dict__)
    sys.modules["main"] = main
    try:
        old = main.VectorIndex()
        old.embeddings = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        old.documents = ["one", "two"]
        old.metadatas = [{"source": "a.pdf", "chunk_id": 0}, {"source": "a.pdf", "chunk_id": 1}]
        with open(tmp_path / "index.pkl", "wb") as f:
            pickle.dump(old, f)
    finally:
        del sys.modules["main"]
    idx = NumpyIndex.load(str(tmp_path))
    assert idx.documents == ["one", "two"] and idx.signature is None
    assert (tmp_path / "index.pkl.migrated").exists() and (tmp_path / "index.json").exists()


def test_legacy_pickle_refuses_other_code(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    with open(tmp_path / "index.pkl", "wb") as f:
        pickle.dump(Evil(), f)
    with pytest.raises(pickle.UnpicklingError):
        NumpyIndex.load(str(tmp_path))


# ── broad questions ("summarise / compare the documents") ─────────────────────
@pytest.mark.parametrize("q", [
    "tell me about those documents in low level",
    "summarise / compare the documents",
    "Compare the two papers",
    "What do these papers have in common?",
    "How do the documents differ?",
    "Summarize each paper",
    "give me an overview",
    "summarise",
    "what is this all about?",
    "What are both papers about?",
])
def test_broad_questions_detected(q):
    assert pipeline.is_broad(q)


@pytest.mark.parametrize("q", [
    "What are the main findings of the SDN paper?",
    "What does the hierarchical framework consist of?",
    "Which datasets were used?",
    "What are its limitations?",
    "Who did the funding acquisition?",
    "Summarise the evaluation results of the round-table protocol in detail",
])
def test_specific_questions_not_broad(q):
    assert not pipeline.is_broad(q)


def test_boilerplate_detection():
    credit = ("Diego Russo: Writing – review & editing, Visualization, Validation, Supervision. "
              "Anna: Funding acquisition, Formal analysis, Data curation, Conceptualization.")
    assert pipeline.is_boilerplate(credit)
    assert pipeline.is_boilerplate("References [1] A. Smith, Agents for security, 2024.")
    assert not pipeline.is_boilerplate("We ran a formal analysis and validation of the software agents.")


class FakeCorpus:
    def __init__(self, docs):
        self.docs = docs            # name -> list of chunk texts
    def keys(self):
        return list(self.docs)
    def search(self, vector, n, key=None):
        names = [key] if key else list(self.docs)
        hits = [{"text": t, "source": k, "page": i + 1, "score": 0.9 - i / 100}
                for k in names for i, t in enumerate(reversed(self.docs[k]))]
        return sorted(hits, key=lambda h: -h["score"])[:n]
    def opening(self, key, n):
        return [{"text": t, "source": key, "page": i + 1, "score": None}
                for i, t in enumerate(self.docs[key][:n])]


def test_broad_question_uses_every_document(ollama):
    corpus = FakeCorpus({
        "a.pdf": ["Abstract A: hierarchical agents.", "Intro A.", "Method A.",
                  "Author contributions: Conceptualization, Data curation, Funding acquisition."],
        "b.pdf": ["Abstract B: SDN for industrial IoT.", "Intro B.", "Results B."],
    })
    prepared = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3",
                                "summarise / compare the documents", [],
                                lambda v, n: corpus.search(v, n), corpus=corpus)
    assert prepared.broad
    texts = [c["text"] for c in prepared.chunks]
    assert texts[:2] == ["Abstract A: hierarchical agents.", "Abstract B: SDN for industrial IoT."]
    assert {c["source"] for c in prepared.chunks} == {"a.pdf", "b.pdf"}
    assert not any("Conceptualization" in t for t in texts)
    assert "The passages come from 2 documents: a.pdf; b.pdf" in prepared.prompt


def test_specific_question_and_single_document_stay_normal(ollama):
    corpus = FakeCorpus({"a.pdf": ["Abstract A."], "b.pdf": ["Abstract B."]})
    client = OllamaClient(ollama.url)
    p = pipeline.prepare(client, Settings(), "llama3", "Which datasets were used?", [],
                         lambda v, n: corpus.search(v, n), corpus=corpus)
    assert not p.broad
    one = FakeCorpus({"a.pdf": ["Abstract A."]})
    p = pipeline.prepare(client, Settings(), "llama3", "summarise the documents", [],
                         lambda v, n: one.search(v, n), corpus=one)
    assert not p.broad
