"""web/app/main.py endpoint functions (FastAPI routing itself is stubbed)."""

import io
import json
import pickle
import sys
import types

import numpy as np
import pytest

from fastapi import BackgroundTasks, HTTPException, UploadFile

BOOK = "The ZenithBook Air 15 has a battery life of 14 hours. " * 5 + \
       "The UltraBook Pro 14 weighs 1.2 kg and has two USB-C ports. " * 5


def upload(web, name, text):
    return web.upload_file(UploadFile(name, io.BytesIO(text.encode())))


def index(web, name):
    bt = BackgroundTasks()
    out = web.index_document(name, bt)
    bt.run()
    return out


def chat(web, question, **kw):
    resp = web.chat_stream(web.ChatRequest(question=question, **kw))
    body = resp.text()
    head, _, rest = body.partition("\n")
    try:
        return json.loads(head), rest
    except ValueError:
        return None, body


def test_upload_index_chat(web, ollama):
    upload(web, "laptops.txt", BOOK)
    index(web, "laptops.txt")
    assert web.index_status("laptops.txt")["status"] == "done"
    ollama.answer = "Fourteen hours."
    head, answer = chat(web, "What is the ZenithBook battery life?")
    assert head["sources"] == ["laptops.txt"] and head["model"] == "llama3"
    assert answer == "Fourteen hours."
    assert web.health()["needs_reindex"] is None


def test_index_is_saved_and_reloaded(web, env):
    upload(web, "a.txt", BOOK)
    index(web, "a.txt")
    web._index = None                                      # simulate a restart
    assert web.list_documents()["documents"][0]["name"] == "a.txt"
    assert (env / "data" / "index.json").exists()


def test_model_list_and_choice(web, ollama):
    models = web.list_models()
    assert [m["name"] for m in models["models"]] == ["llama3", "qwen2.5:7b"]
    assert models["default"] == "llama3"
    upload(web, "a.txt", BOOK)
    index(web, "a.txt")
    head, _ = chat(web, "battery?", model="qwen2.5:7b")
    assert head["model"] == "qwen2.5:7b"
    assert ollama.calls("/api/generate")[-1]["model"] == "qwen2.5:7b"
    with pytest.raises(HTTPException) as e:
        chat(web, "battery?", model="mistral")
    assert e.value.status_code == 400 and "ollama pull mistral" in e.value.detail


def test_models_when_ollama_down(web, monkeypatch):
    web.ollama.base = "http://127.0.0.1:9"
    with pytest.raises(HTTPException) as e:
        web.list_models()
    assert e.value.status_code == 503


def test_stream_error_is_shown(web, ollama):
    upload(web, "a.txt", BOOK)
    index(web, "a.txt")
    ollama.fail_generate_after = 1
    _, answer = chat(web, "battery?")
    assert "[ERROR:" in answer and "model runner crashed" in answer


def test_failed_indexing_reports_error_and_keeps_index(web, ollama):
    upload(web, "a.txt", BOOK)
    index(web, "a.txt")
    upload(web, "b.txt", BOOK + " more")
    ollama.embed_calls_before_failure = 0
    index(web, "b.txt")
    st = web.index_status("b.txt")
    assert st["status"] == "error" and "crashed" in st["error"]
    assert list(web.get_index().sources()) == ["a.txt"]


def test_bad_file_reports_error(web):
    upload(web, "bad.pdf", "not a pdf")
    index(web, "bad.pdf")
    st = web.index_status("bad.pdf")
    assert st["status"] == "error" and "Could not read" in st["error"]
    files = web.list_files()["files"]
    assert files[0]["error"]


def test_reupload_replaces_chunks(web):
    upload(web, "a.txt", BOOK)
    index(web, "a.txt")
    assert upload(web, "a.txt", "Totally different text. " * 10)["replaced"] is True
    assert web.get_index().sources() == {}


def legacy_pickle(folder):
    main = types.ModuleType("main_legacy")
    exec("class VectorIndex:\n    pass", main.__dict__)
    main.VectorIndex.__module__ = "main"
    sys.modules["main"], saved = main, sys.modules.get("main")
    try:
        old = main.VectorIndex()
        old.embeddings = [np.ones(64) * 0.1, np.ones(64) * 0.2]
        old.documents = ["Old chunk about batteries.", "Old chunk about ports."]
        old.metadatas = [{"source": "gone.pdf", "chunk_id": 0}, {"source": "gone.pdf", "chunk_id": 1}]
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "index.pkl", "wb") as f:
            pickle.dump(old, f)
    finally:
        if saved is not None:
            sys.modules["main"] = saved


def test_legacy_index_needs_rebuild_then_works(web, env, ollama):
    legacy_pickle(env / "data")
    web._index = None
    assert "older version" in web.health()["needs_reindex"]
    head, body = chat(web, "battery?")
    assert head is None and "Rebuild index" in body
    upload(web, "a.txt", BOOK)
    index(web, "a.txt")
    assert web.index_status("a.txt")["status"] == "error"   # no mixing old and new vectors
    bt = BackgroundTasks()
    web.reindex(bt)
    bt.run()
    assert web.reindex_status()["status"] == "done"
    assert web.health()["needs_reindex"] is None
    assert web.get_index().sources() == {"gone.pdf": 2}
    index(web, "a.txt")
    assert web.index_status("a.txt")["status"] == "done"


def test_failed_rebuild_keeps_old_index(web, env, ollama):
    legacy_pickle(env / "data")
    web._index = None
    ollama.embed_calls_before_failure = 0
    bt = BackgroundTasks()
    web.reindex(bt)
    bt.run()
    assert web.reindex_status()["status"] == "error"
    web._index = None
    assert len(web.get_index()) == 2 and web.health()["needs_reindex"]


def test_changes_refused_while_rebuilding(web):
    upload(web, "a.txt", BOOK)
    index(web, "a.txt")
    web.rebuild_status["status"] = "rebuilding"
    try:
        for call in (lambda: web.delete_document("a.txt"), lambda: web.delete_file("a.txt"),
                     lambda: upload(web, "b.txt", BOOK), lambda: index(web, "a.txt")):
            with pytest.raises(HTTPException) as e:
                call()
            assert e.value.status_code == 409
    finally:
        web.rebuild_status["status"] = "idle"


def test_followup_rewrite_in_web(web, ollama):
    upload(web, "a.txt", BOOK)
    index(web, "a.txt")
    ollama.answers = ["UltraBook Pro 14 weight", "1.2 kg."]
    head, answer = chat(web, "and its weight?",
                        history=[{"q": "Tell me about the UltraBook Pro 14", "a": "A laptop."}])
    assert head["search_query"] == "UltraBook Pro 14 weight" and answer == "1.2 kg."


def test_path_traversal_still_refused(web):
    with pytest.raises(HTTPException):
        web.upload_path("../../etc/passwd/..")
    assert web.safe_filename("../../evil.txt") == "evil.txt"


def test_broad_question_covers_every_document(web, ollama):
    upload(web, "a.txt", "Abstract. Hierarchical agents for cyber-physical security. " * 10)
    upload(web, "b.txt", "Abstract. SDN and AI for industrial IoT networks. " * 10)
    index(web, "a.txt"); index(web, "b.txt")
    head, _ = chat(web, "tell me about those documents in low level")
    assert head["documents"] == 2
    prompt = ollama.calls("/api/generate")[-1]["prompt"]
    assert "(source: a.txt, abstract)" in prompt and "(source: b.txt, abstract)" in prompt
    assert any(s.startswith("a.txt") for s in head["sources"]) and any(s.startswith("b.txt") for s in head["sources"])
    # Limited to one document with CONTEXT: normal search, no broad mode.
    head, _ = chat(web, "tell me about those documents", filter_source="a.txt")
    assert "documents" not in head and all(s.startswith("a.txt") for s in head["sources"])


def test_reasoning_model_thinking_is_hidden(web, ollama):
    ollama.models.append({"name": "hf.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q8_0-GGUF:latest", "family": "llama",
                          "parameter_size": "8.03B", "capabilities": ["completion"], "context_length": 131072, "size": 1})
    upload(web, "a.txt", BOOK); index(web, "a.txt")
    ollama.answer = "Alright, let's tackle this user query. </think> The battery lasts 14 hours."
    head, answer = chat(web, "battery?", model="hf.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q8_0-GGUF")
    assert head["thinking"] is True
    assert answer == "The battery lasts 14 hours."


def test_sources_numbered_with_passages_and_single_document_filter(web, ollama):
    upload(web, "a.txt", BOOK); index(web, "a.txt")
    upload(web, "b.txt", "Abstract. SDN and AI for industrial IoT networks. " * 30); index(web, "b.txt")
    head, _ = chat(web, "What is the battery life?")
    assert len(head["passages"]) == len(head["sources"])
    assert all(p["text"] and p["cite"] == s for p, s in zip(head["passages"], head["sources"]))
    head, _ = chat(web, "summarise the documents", filter_source="b.txt")
    assert "documents" not in head and all(s.startswith("b.txt") for s in head["sources"])


def blank_pdf(path, pages=2):
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    for _ in range(pages):
        w.add_blank_page(200, 100)
    w.write(str(path))


def test_pdf_page_images(web, env):
    pytest.importorskip("pypdfium2")
    blank_pdf(env / "data" / "uploads" / "paper.pdf")
    resp = web.page_image("paper.pdf", 2, width=400)
    assert resp.media_type == "image/png" and resp.body.startswith(b"\x89PNG")
    assert int.from_bytes(resp.body[16:20], "big") == 400            # IHDR width
    cache = env / "data" / "page_cache"
    assert len(list(cache.rglob("*.png"))) == 1
    assert web.page_image("paper.pdf", 2, width=400).body == resp.body   # served from the cache
    (env / "data" / "uploads" / "notes.txt").write_text("text")
    # The path is reduced to a plain file name, so "../../etc/passwd.pdf" is simply not found.
    for args, codes in [(("paper.pdf", 3), {404}), (("missing.pdf", 1), {404}), (("notes.txt", 1), {400}),
                        (("../../etc/passwd.pdf", 1), {400, 404})]:
        with pytest.raises(HTTPException) as e:
            web.page_image(*args)
        assert e.value.status_code in codes, args
    web.delete_file("paper.pdf")                                     # its page images go too
    assert not list(cache.rglob("*.png"))


def test_passages_carry_file_and_page(web, ollama):
    upload(web, "a.txt", BOOK); index(web, "a.txt")
    head, _ = chat(web, "What is the battery life?")
    assert all(p["source"] == "a.txt" and "page" in p for p in head["passages"])


def test_keyword_search_finds_exact_terms_and_respects_the_filter(web, ollama):
    upload(web, "a.txt", BOOK); index(web, "a.txt")
    upload(web, "mpls.txt", "General routing notes. " * 40 +
           "Labels 0 to 15 are reserved; label 3 is implicit null for penultimate hop popping. " +
           "More general routing notes. " * 40)
    index(web, "mpls.txt")
    idx = web.get_index()
    hits = idx.keyword_search("Which label values are reserved, and what is label 3?", 3)
    assert hits and "implicit null" in hits[0]["text"]
    assert idx.keyword_search("Which label values are reserved?", 3, source_filter="a.txt") == []
    head, _ = chat(web, "Which label values are reserved, and what is label 3 used for?")
    assert any("implicit null" in p["text"] for p in head["passages"])
    web.delete_file("mpls.txt")                               # the keyword index follows changes
    assert web.get_index().keyword_search("label 3 reserved", 3) == []


def test_force_reindex_replaces_an_indexed_file(web):
    upload(web, "book.txt", "LDP i\u2212 the default label di\u2212tribution protocol. " * 30)
    index(web, "book.txt")
    chunks = web.get_index().sources()["book.txt"]
    assert "LDP is the default label distribution protocol" in web.get_index().documents[0]
    bt = BackgroundTasks(); web.index_document("book.txt", bt, force=True); bt.run()
    status = web.index_status("book.txt")
    assert status["status"] == "done" and status["repaired_s"] == 60
    assert web.get_index().sources()["book.txt"] == chunks          # replaced, not duplicated
    assert len(web.get_index()) == chunks


def test_empty_answer_is_reported_as_an_error(web, ollama):
    upload(web, "a.txt", BOOK); index(web, "a.txt")
    ollama.answer = "   "
    head, answer = chat(web, "What is the battery life?")
    assert answer.strip().startswith("[ERROR:") and "returned no answer" in answer
