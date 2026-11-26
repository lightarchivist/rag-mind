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


DAMAGED = ("LDP i\u2212 the default label di\u2212tribution protocol. Label 3 i\u2212 the implicit null "
           "label, re\u2212erved for penultimate hop popping, \u2212uch a\u2212 an OAM me\u2212\u2212age. "
           "Check the acce\u2212\u2212 li\u2212t. ")


def test_broken_s_font_is_repaired_and_watermarks_removed(tmp_path):
    report = {}
    path = write(tmp_path / "book.txt", (DAMAGED + "From the Library of Jane Doe ptg12345678 ") * 10)
    [(_, text)] = extract_pages(path, report)
    assert "LDP is the default label distribution protocol" in text
    assert "reserved for penultimate hop popping, such as an OAM message" in text
    assert "Check the access list." in text
    assert "\u2212" not in text and "From the Library" not in text and "ptg" not in text
    assert report["repaired_s"] == 110 and report["watermarks"] == 10


def test_ligatures_split_words_headings_and_download_stamps(tmp_path):
    report = {}
    text = ("T able 1.Comparative results. T o ensure efﬁciency the ﬂow was checked. V ery good. " * 4 +
            "VIII. C ONCLUSION We conclude. Authorized licensed use limited to: Some University. "
            "Downloaded on November 25,2026 at 15:06:24 UTC from IEEE Xplore. Restrictions apply. End.")
    [(_, out)] = extract_pages(write(tmp_path / "paper.txt", text), report)
    assert "Table 1.Comparative results. To ensure efficiency the flow was checked. Very good." in out
    assert "VIII. CONCLUSION We conclude." in out
    assert "Authorized" not in out and "University" not in out and out.endswith("End.")
    assert report["ligatures"] == 8 and report["stamps"] == 1 and report["split_words"] == 13
    assert [t for _, t in pipeline.captions(out)][0] == "Table 1: Comparative results."


def test_split_capitals_are_joined_only_when_common(tmp_path):
    text = "Antibodies bind T cells. Use a V shaped notch. " + "Normal text here. " * 20
    [(_, out)] = extract_pages(write(tmp_path / "bio.txt", text))
    assert out == text


def test_watermark_without_number_is_removed_when_it_repeats(tmp_path):
    pages = "".join(f"Page {i} text about MPLS.\nFrom the Library of Jane Q Doe\n{400 + i} Cisco Guide\n"
                    for i in range(12))
    report = {}
    [(_, out)] = extract_pages(write(tmp_path / "book.txt", pages + "Sources: From the Library of Congress."), report)
    assert "Jane" not in out and "Doe" not in out and report["watermarks"] == 12
    assert "From the Library of Congress" in out                 # once: ordinary text
    assert "403 Cisco Guide" in out                                # the rest of the line stays
    few = "From the Library of Jane Doe. " * 3 + "Normal text. " * 10
    [(_, out)] = extract_pages(write(tmp_path / "few.txt", few))
    assert out == few


def test_watermark_with_odd_spacing_or_a_running_header_after_it(tmp_path):
    # No-break spaces, the name on the line after "of", and a running header that
    # follows the name on most pages but not all: the name goes from every page.
    pages = []
    for i in range(30):
        mark = "From the\u00a0Library of\nJane\u00a0Q Doe" if i % 2 else "From the Library of Jane Q Doe"
        header = f" Chapter {i}: Labels {500 + i}" if i % 5 else f" {500 + i} Cisco Guide"
        pages.append(f"Page {i} text about MPLS.\n{mark}{header}\nBody of page {i}.\n")
    [(_, out)] = extract_pages(write(tmp_path / "book.txt", "".join(pages)))
    assert "Jane" not in out and "Doe" not in out and "Library" not in out
    assert "Body of page 7." in out and "Labels 507" in out and "505 Cisco Guide" in out


def test_split_capitals_leave_variables_alone(tmp_path):
    text = ("This is a graph. Let G = (V, E) where V is the set of nodes and T is the period. " * 15
            + "T able 1 shows T o be V ery clear and T eachers agree. " * 4 + "The table is very clear.")
    [(_, out)] = extract_pages(write(tmp_path / "paper.txt", text), report := {})
    assert "V is the set of nodes" in out and "T is the period" in out
    assert "Table 1 shows To be Very clear and Teachers agree." in out and report["split_words"] == 16


def test_watermark_names_with_accents_and_particles(tmp_path):
    pages = "".join(f"Page {i} about MPLS.\nFrom the Library of Jos\u00e9 van der Berg\nBody {i}.\n"
                    for i in range(12))
    [(_, out)] = extract_pages(write(tmp_path / "book.txt", pages))
    assert "Jos" not in out and "Berg" not in out and "Body 7." in out


def test_one_unreadable_pdf_page_does_not_lose_the_document(tmp_path, monkeypatch):
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    for _ in range(3):
        w.add_blank_page(100, 100)
    w.write(str(tmp_path / "three.pdf"))
    real = pypdf.PageObject.extract_text
    calls = []

    def flaky(self, *a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("damaged content stream")
        return "Readable page text."
    monkeypatch.setattr(pypdf.PageObject, "extract_text", flaky)
    pages = extract_pages(str(tmp_path / "three.pdf"))
    assert [p for p, _ in pages] == [1, 2, 3] and pages[1][1] == "" and pages[2][1] == "Readable page text."
    monkeypatch.setattr(pypdf.PageObject, "extract_text", lambda self, *a, **k: 1 / 0)
    with pytest.raises(ExtractionError):                 # nothing readable at all: still an error
        extract_pages(str(tmp_path / "three.pdf"))
    monkeypatch.setattr(pypdf.PageObject, "extract_text", real)


def test_real_minus_signs_are_left_alone(tmp_path):
    maths = "With n\u22121 links and x \u2212 y = 3, the cost is small. " * 30   # plenty of real "s"
    [(_, text)] = extract_pages(write(tmp_path / "paper.txt", maths), report := {})
    assert text == maths and report["repaired_s"] == 0
    few = "Only one odd i\u2212 here. " + "Normal sentences with s. " * 5
    [(_, text)] = extract_pages(write(tmp_path / "few.txt", few))
    assert text == few                                  # too few to be the font problem


def test_pdf_page_rendering(tmp_path):
    pytest.importorskip("pypdfium2")
    import zlib
    from ragmind import pages
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    w.add_blank_page(200, 100)
    w.write(str(tmp_path / "p.pdf"))
    png = pages.render(str(tmp_path / "p.pdf"), 1, width=300)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    assert (width, height) == (300, 150)
    with pytest.raises(pages.PageError, match="has no page 2"):
        pages.render(str(tmp_path / "p.pdf"), 2)
    with pytest.raises(pages.PageError, match="Could not open"):
        pages.render(write(tmp_path / "bad.pdf", "not a pdf"), 1)


def test_png_encoder():
    import zlib
    from ragmind import pages
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[1, 2] = (255, 0, 0)
    png = pages.png_bytes(pixels)
    idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
    raw = zlib.decompress(idat)
    assert len(raw) == 2 * (1 + 3 * 3) and raw[-3:] == b"\xff\x00\x00"


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
    def chunk_count(self, key):
        return len(self.docs[key])
    def chunk_range(self, key, start, stop):
        return [{"text": t, "source": key, "page": i + 1, "score": None, "chunk_id": i}
                for i, t in enumerate(self.docs[key]) if start <= i < stop]


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


# ── reasoning models: hide the thinking ───────────────────────────────────────
from ragmind import thinking
from ragmind.ollama import ModelInfo


def run_filter(text, hold, size=3):
    tokens = [text[i:i + size] for i in range(0, len(text), size)]   # tags split across tokens
    return "".join(thinking.visible(tokens, hold=hold))


@pytest.mark.parametrize("hold", [True, False])
def test_think_block_is_hidden(hold):
    assert run_filter("<think>Let me reason about it.</think>\n\nThe answer.", hold) == "The answer."


def test_prefilled_template_lone_closing_tag():
    text = "Alright, let's tackle this user query.\n</think>\n\nBased on the passages, X."
    assert run_filter(text, hold=True) == "Based on the passages, X."


def test_normal_answers_pass_through():
    assert run_filter("Plain answer with <b>no</b> thinking.", hold=False) == "Plain answer with <b>no</b> thinking."
    assert run_filter("  < 5 items", hold=False) == "  < 5 items"


def test_unclosed_reasoning_is_shown_rather_than_lost():
    assert run_filter("Answer without any tags.", hold=True) == "Answer without any tags."
    assert run_filter("<think>cut off before closing", hold=False) == "cut off before closing"


def test_which_models_need_filtering():
    hf = ModelInfo("hf.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q8_0-GGUF:latest", capabilities=["completion"])
    assert thinking.reasons_unseparated(hf)
    assert thinking.reasons_unseparated(ModelInfo("deepseek-r1:8b"))
    assert not thinking.reasons_unseparated(ModelInfo("deepseek-r1:8b", capabilities=["completion", "thinking"]))
    assert not thinking.reasons_unseparated(ModelInfo("gpt-oss:20b", capabilities=["completion", "thinking"]))
    assert not thinking.reasons_unseparated(ModelInfo("llama3:latest", capabilities=["completion"]))


def test_rewrite_ignores_reasoning(ollama):
    ollama.models.append({"name": "deepseek-r1:8b", "family": "qwen2", "parameter_size": "8B",
                          "capabilities": ["completion"], "context_length": 32768, "size": 1})
    ollama.answers = ["Okay, the user asks about weight.\n</think>\nWhat is the weight of the UltraBook Pro 14?"]
    q = pipeline.rewrite_followup(OllamaClient(ollama.url), "deepseek-r1:8b", "and its weight?",
                                  [{"q": "Tell me about the UltraBook Pro 14", "a": "A laptop."}], 4096)
    assert q == "What is the weight of the UltraBook Pro 14?"
    assert ollama.calls("/api/generate")[-1]["options"]["num_predict"] == 1024


# ── follow-up rewriting only when needed ──────────────────────────────────────
@pytest.mark.parametrize("q", ["and its weight?", "What are its limitations?", "and charging?",
                               "What about the second one?", "Why?", "How do they compare to that?"])
def test_followups_detected(q):
    assert pipeline.looks_like_followup(q)


@pytest.mark.parametrize("q", [
    "How does the round-table protocol resolve conflicts between subsystem supervisors?",
    "Which datasets were used to evaluate the hierarchical framework?",
    "What scheduling mechanisms are used in time-sensitive networking?",
])
def test_complete_questions_not_rewritten(q):
    assert not pipeline.looks_like_followup(q)


def test_complete_question_after_history_is_searched_as_asked(ollama):
    q = "How does the round-table protocol resolve conflicts between subsystem supervisors?"
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3", q,
                         [{"q": "summarise the documents", "a": "Two papers."}], lambda v, n: [])
    assert p.search_query == q
    assert not [b for b in ollama.calls("/api/generate") if not b.get("stream", True)]   # no rewrite call


def test_drifting_rewrite_still_finds_original_passages(ollama):
    ollama.answers = ["How do specialised agents assess security posture?"]
    calls = []

    def search(vector, n):
        calls.append(vector)
        if len(calls) == 1:     # search with the (drifted) rewrite
            return [{"text": "Agents assess posture.", "source": "a.pdf", "page": 2, "score": 0.6}]
        return [{"text": "Round-table protocol details.", "source": "a.pdf", "page": 11, "score": 0.8}]

    p = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3", "and how does it resolve conflicts?",
                         [{"q": "Tell me about the round-table protocol", "a": "It is a discussion."}], search)
    assert len(calls) == 3              # rewrite, original wording, previous answer
    assert [c["page"] for c in p.chunks] == [11, 2]


def test_passage_labels_have_no_scores():
    prompt, _, _ = build_prompt("Q?", [chunk(1)], [], 8192)
    assert "relevance" not in prompt and "[1] (source: a.pdf, p. 1)" in prompt



# ── summaries: abstract + conclusion; history only for follow-ups ─────────────
def paper(name, findings):
    chunks = [f"{name}. J. Smith a,*, A. Rossi b. a University of Naples, Department of Engineering. "
              "E-mail: smith@unina.it. Received 3 May 2025; accepted 9 June 2025. doi 10.1016/x",
              f"A B S T R A C T We study {name}. Our experiments show {findings}.",
              f"1. Introduction {name} matters for industrial systems."]
    chunks += [f"Section {i} discusses implementation details of {name}." for i in range(3, 40)]
    chunks += [f"Further discussion of limitations. 6. Conclusions This paper presented {name}. "
               f"The main result is that {findings}.",
               "Future work will extend the evaluation to more datasets.",
               "References [1] A. Author, Some title, 2024. [2] B. Author, Other title, 2023."]
    return chunks


def test_summary_includes_abstract_and_conclusion_not_title_page(ollama):
    corpus = FakeCorpus({"a.pdf": paper("hierarchical agents", "recall rises 12% with round-table"),
                         "b.pdf": paper("AI-enabled SDN", "latency drops by 30%")})
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3", "Summarise the key findings", [],
                         lambda v, n: corpus.search(v, n), corpus=corpus)
    assert p.broad
    texts = " ".join(c["text"] for c in p.chunks)
    assert "recall rises 12% with round-table" in texts and "latency drops by 30%" in texts
    assert texts.count("6. Conclusions This paper presented") == 2
    assert "smith@unina.it" not in texts and "References [1]" not in texts
    assert "(source: a.pdf, p. 2, abstract)" in p.prompt and "conclusion)" in p.prompt


@pytest.mark.parametrize("q,expected", [
    ("Summarise the key findings", False),
    ("tell me about those documents in low level", False),
    ("What do these papers have in common?", False),
    ("summarise it", True),
    ("Explain the second point in more detail", True),
    ("Why does the framework use a round-table discussion between supervisors?", False),
])
def test_followup_detection_with_broad_questions(q, expected):
    assert pipeline.looks_like_followup(q) is expected


def test_rewrite_rejects_list_items_and_echoed_answers(ollama):
    client = OllamaClient(ollama.url)
    history = [{"q": "Tell me about the papers", "a": "1. *Hierarchical framework* ..."}]
    for reply in ["1. *Hierarchical Multi-Agent AI Framework for Cybersecurity",
                  "- the second paper",
                  "Line one\nLine two\nLine three\nLine four"]:
        ollama.answers = [reply]
        assert pipeline.rewrite_followup(client, "llama3", "and its limits?", history, 4096) == "and its limits?"


def test_history_only_for_followups(ollama):
    client = OllamaClient(ollama.url)
    history = [{"q": "Summarise the documents", "a": "Both stress traceability (invented claim)."}]
    complete = pipeline.prepare(client, Settings(), "llama3", "Summarise the key findings", history,
                                lambda v, n: [])
    assert "CONVERSATION SO FAR" not in complete.prompt and "invented claim" not in complete.prompt
    ollama.answers = ["What are the limitations of the framework?"]
    follow = pipeline.prepare(client, Settings(), "llama3", "and its limitations?", history, lambda v, n: [])
    assert "CONVERSATION SO FAR" in follow.prompt


# ── 2.2: document layout, back matter, following text ─────────────────────────
def structured_paper():
    """Laid out like the Hierarchical paper (chunk by chunk)."""
    c = ["Journal title page. a University of Naples. E-mail: x@unina.it. Received 2025. doi 10.1/x",
         "Keywords Network security Agentic AI A B S T R A C T Cyber-physical systems need security.",
         "This paper proposes a hierarchical multi-agent framework.",
         "Specialized agents analyse data; supervisors aggregate findings.",
         "The framework is evaluated using four LLMs across three datasets (PicoDomain, CERT r5.2, and SWaT),",
         "reaching an accuracy of 90.18% on PicoDomain and 96.97% on SWaT.",
         "and a foundation for AI security. 1. Introduction Cyber-physical systems are everywhere."]
    c += [f"Method details part {i}, see Appendix A.2 for the prompts." for i in range(7, 40)]
    c += ["Earlier results. 6. Conclusion, limitations and future work This work introduced a framework.",
          "Table 6 Agreement 0.41 0.29 0.47 0.52 0.61 0.38 0.64 0.71",
          "Experiments across three datasets show perfect recall for models of 8B and above.",
          "Limitations include prompt injection risks.",
          "CRediT authorship contribution statement A: Conceptualization, Data curation, Funding acquisition.",
          "Appendix A. Agent prompts This appendix lists the round-table prompt in full.",
          "You are a subsystem supervisor. Share your assessment in the round-table discussion.",
          "References [1] A. Smith, FWOS scheduling for TSN, 2021. [2] B. Lee, Agents, 2023.",
          "[3] C. Wu, Round-table deliberation in security systems, 2022."]
    return c


def test_layout_finds_whole_abstract_conclusion_appendix_and_references():
    corpus = FakeCorpus({"p.pdf": structured_paper()})
    lay = pipeline.doc_layout(corpus, "p.pdf")
    assert [c["chunk_id"] for c in lay["abstract"]] == [1, 2, 3, 4, 5, 6]   # up to the introduction
    assert "96.97% on SWaT" in " ".join(c["text"] for c in lay["abstract"])
    concl = " ".join(c["text"] for c in lay["conclusion"])
    assert "This work introduced" in concl and "perfect recall" in concl and "CRediT" not in concl
    assert lay["appendix_start"] == 45          # the heading, not the early "see Appendix A.2"
    assert lay["references_start"] == 47


def test_broad_questions_skip_appendix_and_references(ollama):
    paper = structured_paper()
    corpus = FakeCorpus({"a.pdf": paper, "b.pdf": paper[:40] + ["7. Conclusion Results for b."]})
    corpus.search = lambda v, n, key=None: [
        {"text": t, "source": key, "page": 1, "score": 0.9, "key": key, "chunk_id": i}
        for i, t in enumerate(corpus.docs[key]) if "round-table" in t or "FWOS" in t][:n]
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3", "Summarise the key findings", [],
                         lambda v, n: [], corpus=corpus)
    texts = " ".join(c["text"] for c in p.chunks)
    assert "96.97% on SWaT" in texts and "perfect recall" in texts
    assert "You are a subsystem supervisor" not in texts and "FWOS" not in texts


def test_normal_questions_skip_reference_list_and_extend_passages(ollama):
    paper = structured_paper()
    corpus = FakeCorpus({"a.pdf": paper})
    hits = lambda v, n: [
        {"text": paper[i], "source": "a.pdf", "page": 1, "score": 0.9 - i / 1000, "key": "a.pdf", "chunk_id": i}
        for i in (47, 4)]
    client = OllamaClient(ollama.url)
    p = pipeline.prepare(client, Settings(), "llama3", "How is TSN scheduling improved?", [], hits, corpus=corpus)
    texts = [c["text"] for c in p.chunks]
    assert not any("FWOS" in t for t in texts)                     # reference list left out
    assert any("PicoDomain" in t and "96.97% on SWaT" in t for t in texts)   # chunk 4 + chunk 5
    p = pipeline.prepare(client, Settings(), "llama3", "Which references are cited about TSN?", [],
                         hits, corpus=corpus)
    assert any("FWOS" in c["text"] for c in p.chunks)              # unless asked for


def test_merge_overlapping_chunks():
    assert pipeline.merge_overlapping("we adopt three datasets. We first", "datasets. We first evaluate X") \
        == "we adopt three datasets. We first evaluate X"
    assert pipeline.merge_overlapping("abc", "xyz") == "abc xyz"


# ── passage context: tables, lead-in, title page ──────────────────────────────

def tables_paper(captions_above=True):
    cert = "Qwen 4B 71.20% 64.10% 85.30% 73.20% 38.40% Llama 8B 90.10% 84.50% 100% 91.60% 17.20%"
    swat = "Qwen 4B 75.40% 14.80% 70.20% 24.40% 20.90% Llama 8B 95.80% 55.30% 100% 71.20% 3.40%"
    t3, t4 = "Table 3 Performance on the CERT dataset. Best in bold.", "Table 4 Performance on the SWaT dataset."
    if captions_above:
        return ["Results are summarised as shown in Table 3 below.", t3, "Model Acc Prec Rec F1 FPR",
                cert + " " + t4, swat, "The discussion of these results follows."]
    return ["Results are summarised as shown in Table 3 below.", "Model Acc Prec Rec F1 FPR",
            cert, t3 + " Model Acc Prec Rec F1 FPR " + swat, t4, "The discussion follows."]


def test_table_rows_are_labelled_with_their_own_caption():
    corpus = FakeCorpus({"a.pdf": tables_paper()})
    assert pipeline.caption_placement(corpus, "a.pdf") == "above"
    passage = dict(corpus.chunk_range("a.pdf", 3, 4)[0], key="a.pdf")
    [out] = pipeline.label_tables(corpus, pipeline.with_following_text(corpus, [passage]))
    assert out["text"].startswith("[The table rows in this passage before any new caption are from "
                                  "Table 3: Performance on the CERT dataset.]")
    assert "17.20%\n\nTable 4 Performance on the SWaT dataset." in out["text"]
    # "as shown in Table 3 below" is a reference, not a caption
    assert [t for _, t in pipeline.captions(" ".join(tables_paper()))] == [
        "Table 3: Performance on the CERT dataset.", "Table 4: Performance on the SWaT dataset."]
    # Plain text is left alone.
    prose = dict(corpus.chunk_range("a.pdf", 5, 6)[0], key="a.pdf")
    assert pipeline.label_tables(corpus, [prose])[0]["text"] == "The discussion of these results follows."


def test_book_style_tables_and_command_output():
    text = ("as listed in Table 5-3. Table 5-3 MPLS Label Ranges Platform Range IOS 16 1048575 "
            "Table 5.4: LDP Timers Hello 5 15 Holdtime 15 45")
    assert [t.split(":")[0] for _, t in pipeline.captions(text)] == ["Table 5-3", "Table 5.4"]
    routes = ("O 10.1.1.0/24 [110/2] via 10.0.0.2, 00:01:02, Gi0/0 O 10.1.2.0/24 [110/3] via 10.0.0.6, "
              "00:01:05, Gi0/1 ip address 192.168.1.1 255.255.255.0 neighbor 10.0.0.1 remote-as 65000")
    assert not pipeline.has_table_rows(routes)
    assert pipeline.has_table_rows("Qwen 4B 71.2% 64.1% 85.3% 73.2% 38.4% Llama 8B 90.1% 84.5% 100%")
    assert pipeline.has_table_rows("Region 1,024 2,048 4,096 8,192 16,384 32,768")


def test_captions_below_tables_label_rows_with_the_next_caption():
    corpus = FakeCorpus({"b.pdf": tables_paper(captions_above=False)})
    assert pipeline.caption_placement(corpus, "b.pdf") == "below"
    passage = dict(corpus.chunk_range("b.pdf", 2, 3)[0], key="b.pdf")
    [out] = pipeline.label_tables(corpus, [passage])
    assert "are from Table 3: Performance on the CERT dataset.]" in out["text"]


def test_lead_in_keeps_who_did_the_work(ollama):
    prev = ("Background on TSN. For instance, in Ref. [12], the authors study queues in TSN devices. "
            "The authors have proved the queue scheduling")
    cur = "proved the queue scheduling is NP-hard. By formulating it with SMT they propose two heuristics."
    assert pipeline.lead_in(prev, cur) == ("For instance, in Ref. [12], the authors study queues in TSN "
                                           "devices. The authors have")
    assert pipeline.lead_in("Short.", "Next sentence.") == "Short."
    corpus = FakeCorpus({"s.pdf": ["Title page of s.", prev, cur]})
    hits = lambda v, n: [{"text": cur, "source": "s.pdf", "page": 3, "score": 0.9, "key": "s.pdf",
                          "chunk_id": 2}]
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3",
                         "How is TSN scheduling improved?", [], hits, corpus=corpus)
    assert p.chunks[0]["text"].startswith("For instance, in Ref. [12], the authors")


def test_author_questions_include_the_title_page(ollama):
    for q in ("Who are the authors of the SDN paper, and where are they based?",
              "Who wrote the hierarchical paper?", "List the authors' affiliations"):
        assert pipeline.asks_about_authors(q), q
    for q in ("How do the authors improve scheduling?", "What do the authors conclude?"):
        assert not pipeline.asks_about_authors(q), q
    corpus = FakeCorpus({
        "Hierarchical multi-agent AI framework.pdf": ["Hierarchical framework. A. Rossi, University of Turin.",
                                                      "A B S T R A C T Agents.", "Method."],
        "How AI-enabled SDN technologies improve IIoT.pdf": ["How AI-enabled SDN... L. Chen, Example University",
                                                             "c Research Council, Cape Town, South Africa",
                                                             "ABSTRACT SDN for IIoT.", "Body."]})
    hits = lambda v, n: [{"text": "Body.", "source": "x", "page": 4, "score": 0.3,
                          "key": "How AI-enabled SDN technologies improve IIoT.pdf", "chunk_id": 3}]
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3",
                         "Who are the authors of the SDN paper, and where are they based?", [], hits,
                         corpus=corpus)
    texts = " ".join(c["text"] for c in p.chunks)
    assert "Example University" in texts and "Cape Town" in texts and "Rossi" not in texts
    assert p.chunks[0]["section"] == "title page"


def test_citation_brackets_and_grouped_sources():
    from ragmind.prompt import group_sources, tidy_citations
    assert tidy_citations("accuracy【9】【11】") == "accuracy[9][11]"
    chunks = [{"source": "a.pdf", "page": 1, "section": "abstract", "text": "x"},
              {"source": "b.pdf", "page": 1, "section": "abstract", "text": "y"},
              {"source": "a.pdf", "page": 1, "section": "abstract", "text": "z"},
              {"source": "a.pdf", "page": 9, "section": "", "text": "w"}]
    assert group_sources(chunks) == [([1, 3], "a.pdf, p. 1", "abstract"), ([2], "b.pdf, p. 1", "abstract"),
                                     ([4], "a.pdf, p. 9", "")]


def test_that_counts_only_when_it_points_back():
    for q in ("Explain that", "Is that true?", "How does that work in IOS XR?",
              "Tell me more about that approach", "What about that one?"):
        assert pipeline.looks_like_followup(q), q
    for q in ("What does TI-LFA provide that classic LFA does not?",
              "Which routers run the protocol that carries labels between PEs?",
              "Why does the book say that SR removes the need for LDP?"):
        assert not pipeline.looks_like_followup(q), q


# ── hybrid search, books ──────────────────────────────────────────────────────

def test_keyword_index_needs_distinctive_words():
    from ragmind.keyword import KeywordIndex, tokens
    assert "ti-lfa" in tokens("What does TI-LFA do?") and "lfa" in tokens("TI-LFA")
    texts = [f"General routing text number {i} about networks." for i in range(40)]
    texts += ["Labels 0 to 15 are reserved; label 3 is implicit null.",
              "The book covers TI-LFA and classic LFA fast reroute."]
    ki = KeywordIndex(texts)
    assert ki.search("Which label values are reserved, and what is label 3?", 3)[0][0] == 40
    assert ki.search("How does TI-LFA compare with LFA?", 3)[0][0] == 41
    assert ki.search("What is the capital of France?", 3) == []
    assert ki.search("routing networks", 3) == []          # only common words: no keyword hits
    assert ki.search("reserved labels", 3, allowed={41}) == []


def test_fusion_keeps_keyword_hits_and_drops_weak_similarity_only_ones():
    sim = [{"text": "close topic", "score": 0.6}, {"text": "weak", "score": 0.1},
           {"text": "both", "score": 0.5}]
    kw = [{"text": "exact terms", "score": None}, {"text": "both", "score": None}]
    out = [c["text"] for c in pipeline.fuse(sim, kw, 3, 0.25)]
    assert out[0] == "both" and set(out) == {"both", "close topic", "exact terms"}


def test_phrases_count_and_one_rare_word_is_not_enough():
    from ragmind.keyword import KeywordIndex, query_groups
    assert {"label 3"} in query_groups("What is label 3 used for?")
    texts = [f"MPLS label distribution text {i}." for i in range(60)]
    texts += ["Copyright 2024. All rights reserved.",                        # 60
              "Reserved Instances lower the cost of EC2.",                   # 61
              "Labels 0-15 are reserved; label 3 is implicit null (PHP)."]   # 62
    ki = KeywordIndex(texts)
    hits = [i for i, _ in ki.search("Which MPLS label values are reserved, and what is label 3 used for?", 5)]
    assert hits == [62]                       # "reserved" alone doesn't bring in 60 or 61
    assert ki.search("Is anything reserved?", 5) == []


def test_keyword_hits_add_to_similarity_results_but_never_push_out_the_best():
    sim = [{"text": f"similar {i}", "score": 0.9 - i / 100} for i in range(12)]
    sim.append({"text": "table 15-2", "score": 0.84})          # similarity rank 7 of 13
    kw = [{"text": f"keyword {i}", "score": None} for i in range(5)] + [{"text": "similar 1", "score": None}]
    out = [c["text"] for c in pipeline.fuse(sim, kw, 6, 0.25)]
    assert out[0] == "similar 1"                               # found by both: first
    assert out[1:4] == ["similar 0", "similar 2", "similar 3"]  # best similarity results kept
    assert out[4:] == ["keyword 0", "keyword 1"]               # at most two keyword-only additions
    few = [{"text": "only one", "score": 0.5}, {"text": "too weak", "score": 0.1}]
    out = [c["text"] for c in pipeline.fuse(few, kw[:4], 6, 0.25)]
    assert out == ["only one", "keyword 0", "keyword 1", "keyword 2", "keyword 3"]


def test_prepare_uses_keyword_search_when_the_corpus_has_it(ollama):
    corpus = FakeCorpus({"a.pdf": ["Something about routing."] * 3 +
                                  ["Labels 0 to 15 are reserved; label 3 is implicit null."]})
    corpus.keyword_search = lambda q, n: [dict(corpus.chunk_range("a.pdf", 3, 4)[0], key="a.pdf")]
    similar = lambda v, n: [dict(corpus.chunk_range("a.pdf", 0, 1)[0], key="a.pdf", score=0.9)]
    client = OllamaClient(ollama.url)
    p = pipeline.prepare(client, Settings(), "llama3", "What is label 3?", [], similar, corpus=corpus)
    assert any("implicit null" in c["text"] for c in p.chunks)
    p = pipeline.prepare(client, Settings(keyword_search=False), "llama3", "What is label 3?", [],
                         similar, corpus=corpus)
    assert not any("implicit null" in c["text"] for c in p.chunks)


def book():
    front = ["Cover. Copyright 2024 Pearson. ISBN 978-0-00. All rights reserved.",
             "Register your book to access the companion website, practice test software and "
             "the study planner. Use the access code inside.",
             "Contents Chapter 1 MPLS Chapter 2 Segment Routing"]
    body = [f"Chapter text {i}: MPLS forwarding and label distribution explained." for i in range(300)]
    tail = ["Summary and Conclusions of the exam guide.", "Index A ABR B BGP"]
    return front + body + tail


def test_books_get_no_abstract_and_publisher_pages_are_skipped(ollama):
    corpus = FakeCorpus({"guide.pdf": book(), "notes.txt": ["Short notes on LDP timers.", "More notes."]})
    lay = pipeline.doc_layout(corpus, "guide.pdf")
    assert lay["book"] and lay["abstract"] == [] and lay["conclusion"] == []
    short = pipeline.doc_layout(corpus, "notes.txt")
    assert not short["book"] and short["abstract"][0]["text"] == "Short notes on LDP timers."
    assert pipeline.is_boilerplate(book()[1])
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3", "Summarise the key findings", [],
                         lambda v, n: [], corpus=corpus)
    texts = " ".join(c["text"] for c in p.chunks)
    assert "companion website" not in texts and "Short notes" in texts


def test_rewrite_keeps_descriptions_unnamed(ollama):
    history = [{"q": "Which model performed best?", "a": "Model B on dataset X [1]."}]
    client = OllamaClient(ollama.url)
    ollama.answers = ["How did the smallest model, Model B, compare to the others on dataset X?"]
    assert pipeline.rewrite_followup(client, "llama3", "And how did the smallest model compare?",
                                     history, 4096) == \
        "How did the smallest model compare to the others on dataset X?"
    ollama.answers = ["How did Model B compare on dataset X?"]          # description replaced
    assert pipeline.rewrite_followup(client, "llama3", "And how did the smallest model compare?",
                                     history, 4096) == "And how did the smallest model compare?"
    assert "do not replace them with a name" in ollama.calls("/api/generate")[-1]["prompt"]


def test_rewrite_gets_room_for_models_that_think(ollama):
    ollama.models.append({"name": "gpt-oss:20b", "family": "gptoss", "parameter_size": "20.9B",
                          "capabilities": ["completion", "thinking"], "context_length": 131072, "size": 1})
    ollama.answers = ["What is the weight of the UltraBook Pro 14?"]
    pipeline.rewrite_followup(OllamaClient(ollama.url), "gpt-oss:20b", "and its weight?",
                              [{"q": "Tell me about the UltraBook Pro 14", "a": "A laptop."}], 4096)
    assert ollama.calls("/api/generate")[-1]["options"]["num_predict"] == 1024


def test_prompt_rules_and_numbered_passages():
    prompt, _, _ = build_prompt("Q?", [chunk(1), chunk(2)], [], 8192)
    assert "[2]" in prompt and "name every one" in prompt and "Ref. [89]" in prompt
    assert "Check the question's assumptions" in prompt
