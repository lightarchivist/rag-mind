"""Document structure (titles, section headings) and the retrieval that uses it:
named documents, depth and breadth questions, stemming, follow-up detection."""

import pytest

from ragmind import pipeline, structure
from ragmind.chunking import chunk_pages
from ragmind.config import Settings
from ragmind.keyword import KeywordIndex, stem
from ragmind.ollama import OllamaClient
from ragmind.prompt import build_prompt, documents_note, format_passage, group_sources, place

FILLER = "The quartz relay network was measured over several weeks in the field. "

PAPER = [
    (1, "Lattice Routing for Quiet Meadows\nAda Brightwater, Tom Q. Fenwick\n"
        "Abstract\n" + FILLER * 3),
    (2, "Proceedings of Imaginary Systems 12 (2031)\n1. Introduction\n" + FILLER * 6 +
        "\n2. Method\n" + FILLER * 4 + "\n2.1. Sensor placement\n" + FILLER * 8),
    (3, "Proceedings of Imaginary Systems 12 (2031)\n2.2. Calibration of the relays\n" + FILLER * 8 +
        "\n3. Results\n" + FILLER * 6 + "\n1. A numbered list item that is not a heading\n"),
    (4, "Proceedings of Imaginary Systems 12 (2031)\n4. Conclusion and future work\n" + FILLER * 4 +
        "\nAppendix A. Relay settings\n1. Set the gain\n2. Set the offset\nReferences\n[1] A. Nobody, 2030."),
]


def test_text_headings_follow_the_numbering_and_skip_running_headers():
    headings = [(p, level, title) for p, level, title in structure.text_headings(PAPER)]
    titles = [t for _, _, t in headings]
    assert titles == ["Abstract", "1. Introduction", "2. Method", "2.1. Sensor placement",
                      "2.2. Calibration of the relays", "3. Results", "4. Conclusion and future work",
                      "Appendix A. Relay settings", "References"]
    assert all("Proceedings" not in t for t in titles)
    assert dict((t, level) for _, level, t in headings)["2.1. Sensor placement"] == 2


def test_chunks_get_the_section_they_are_in():
    chunks = chunk_pages(PAPER, 300, 40)
    title = structure.analyse("lattice.txt", PAPER, chunks)
    assert title == "Lattice Routing for Quiet Meadows"
    by_page = {}
    for ch in chunks:
        by_page.setdefault(ch.page, set()).add(ch.heading)
    assert "2. Method › 2.2. Calibration of the relays" in by_page[3]
    assert any(h == "4. Conclusion and future work" for h in by_page[4])
    assert structure.deepest("2. Method › 2.2. Calibration") == "2.2. Calibration"
    assert structure.section_of("Part I › Chapter 3 › 3.4 Timers") == "Part I › Chapter 3"


def test_a_broken_structure_never_stops_indexing(monkeypatch):
    chunks = chunk_pages(PAPER, 300, 40)
    monkeypatch.setattr(structure, "text_headings", lambda pages: 1 / 0)
    assert structure.analyse("lattice.txt", PAPER, chunks) == "Lattice Routing for Quiet Meadows"
    assert all(ch.heading == "" for ch in chunks)


def test_title_falls_back_to_the_file_name():
    assert structure.document_title("notes-2031.txt", [(None, "x = 3\n42\n")]) == "notes-2031"


def test_light_stemming_joins_word_forms():
    assert stem("chunks") == stem("chunked") == stem("chunking") == stem("chunk")
    assert stem("policies") == stem("policy")
    assert stem("embedding") == stem("embedded") == stem("embed")
    assert stem("addresses") == stem("address")
    assert stem("ti-lfa") == "ti-lfa" and stem("ospfv3") == "ospfv3" and stem("status") == "status"
    index = KeywordIndex(["The corpus was split into chunks of 512 tokens with an overlap of 50 tokens.",
                         "Unrelated text about meadows.", "More unrelated text about relays."])
    assert [i for i, _ in index.search("How was the knowledge base chunked and what overlap?", 3)] == [0]


TITLES = {
    "brightwater-2031.pdf": "Lattice Routing for Quiet Meadows",
    "fenwick.pdf": "Trust-Weighted Lattice Agents: A Hybrid Framework for Calm Networks",
    "orchard-guide.pdf": "Orchard Networking Official Cert Guide",
    "harbor-guide.pdf": "Harbor Security Official Cert Guide",
}


@pytest.mark.parametrize("question, expected", [
    ("What gain did the quiet meadows paper use?", ["brightwater-2031.pdf"]),
    ("What are the signals in the Trust-Weighted framework, and how are they combined?", ["fenwick.pdf"]),
    ("Who are the authors of the Brightwater paper?", ["brightwater-2031.pdf"]),
    ("How do the quiet meadows paper and the Trust-Weighted paper differ?",
     ["brightwater-2031.pdf", "fenwick.pdf"]),
    ("Compare the meadows and calm networks papers", ["brightwater-2031.pdf", "fenwick.pdf"]),
    ("Summarise the lattice papers", ["brightwater-2031.pdf", "fenwick.pdf"]),
    ("In the whole Orchard book, where are timers discussed?", ["orchard-guide.pdf"]),
    ("In the paper on quiet meadows, what was measured?", ["brightwater-2031.pdf"]),
    ("How does the official cert guide describe it?", []),      # two cert guides: names neither
    ("What does the lattice paper say?", []),                    # two lattice papers, singular
    ("What does the proposed framework do?", []),
    ("What is the role of the relay in the system?", []),
])
def test_documents_named_by_title_or_file_name(question, expected):
    assert pipeline.documents_named(question, TITLES) == expected


def test_pronoun_in_a_second_clause_is_not_a_followup():
    assert not pipeline.looks_like_followup(
        "What are the three signals in the Trust-Weighted framework, and how are they combined?")
    assert not pipeline.looks_like_followup("Which datasets were used in the evaluation, and how big are they?")
    assert pipeline.looks_like_followup("How are they combined?")
    assert pipeline.looks_like_followup("What did they find, and how?")
    assert pipeline.looks_like_followup("What are the limits of it, and how do they matter?")


def test_passages_show_their_section_and_documents_their_titles():
    ch = {"text": "Gain is 3 dB.", "source": "b.pdf", "page": 4, "heading": "2. Method › 2.2. Calibration",
          "title": "Lattice Routing for Quiet Meadows"}
    assert format_passage(1, ch) == "[1] (source: b.pdf, p. 4, in 2. Method › 2.2. Calibration)\nGain is 3 dB."
    assert place(ch) == "2.2. Calibration"
    assert place(dict(ch, heading="Abstract", section="abstract")) == "Abstract"
    assert place(dict(ch, heading="", section="conclusion")) == "conclusion"
    assert documents_note([ch]) == 'Document titles: b.pdf = "Lattice Routing for Quiet Meadows".'
    assert documents_note([dict(ch, title="b")]) == ""          # a title that is just the file name
    prompt, _, _ = build_prompt("Q?", [ch], [], 8192)
    assert 'b.pdf = "Lattice Routing for Quiet Meadows"' in prompt
    two = [ch, dict(ch, heading="3. Results")]
    assert [g[2] for g in group_sources(two)] == ["2.2. Calibration", "3. Results"]


class Corpus:
    """Documents of (heading, text) chunks."""

    def __init__(self, docs, titles=None):
        self.docs, self._titles = docs, titles or {}

    def keys(self):
        return list(self.docs)

    def titles(self):
        return {k: self._titles.get(k, k) for k in self.docs}

    def _row(self, key, i, score=None):
        heading, text = self.docs[key][i]
        return {"text": text, "source": key, "page": i + 1, "score": score, "key": key,
                "chunk_id": i, "heading": heading, "title": self._titles.get(key, "")}

    def search(self, vector, n, key=None):
        keys = [key] if key else list(self.docs)
        rows = [self._row(k, i, 0.9 - i / 100) for k in keys for i in range(len(self.docs[k]))]
        return sorted(rows, key=lambda r: -r["score"])[:n]

    def chunk_count(self, key):
        return len(self.docs[key])

    def chunk_range(self, key, start, stop):
        return [self._row(key, i) for i in range(max(start, 0), min(stop, len(self.docs[key])))]


def test_a_named_document_limits_the_search(ollama):
    corpus = Corpus({"a.pdf": [("1. Intro", f"Meadow fact {i}.") for i in range(10)],
                     "b.pdf": [("1. Intro", f"Harbor fact {i}.") for i in range(10)]},
                    {"a.pdf": "Lattice Routing for Quiet Meadows", "b.pdf": "Harbor Security Notes"})
    calls = []

    def search(vector, n):
        calls.append(n)
        return corpus.search(vector, n)

    p = pipeline.prepare(OllamaClient(ollama.url), Settings(keyword_search=False), "llama3",
                         "What does the quiet meadows paper measure?", [], search, corpus=corpus)
    assert {c["source"] for c in p.chunks} == {"a.pdf"} and not calls
    assert "The question names one document: a.pdf" in p.prompt


def test_two_named_documents_each_get_passages(ollama):
    corpus = Corpus({"a.pdf": [("", f"Meadow fact {i}.") for i in range(10)],
                     "b.pdf": [("", f"Harbor fact {i}.") for i in range(10)],
                     "c.pdf": [("", f"Orchard fact {i}.") for i in range(10)]},
                    {"a.pdf": "Lattice Routing for Quiet Meadows", "b.pdf": "Harbor Security Notes",
                     "c.pdf": "Orchard Notes"})
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(keyword_search=False), "llama3",
                         "How do the quiet meadows paper and the harbor paper differ in calibration?",
                         [], lambda v, n: corpus.search(v, n), corpus=corpus)
    sources = [c["source"] for c in p.chunks]
    assert set(sources) == {"a.pdf", "b.pdf"} and sources[:2] == ["a.pdf", "b.pdf"]
    assert "The question names 2 documents" in p.prompt


def test_summary_of_one_named_document_stays_in_that_document(ollama):
    corpus = Corpus({"a.pdf": [("Abstract", "Abstract: meadows."), ("1. Intro", "Intro A.")],
                     "b.pdf": [("Abstract", "Abstract: harbors."), ("1. Intro", "Intro B.")]},
                    {"a.pdf": "Lattice Routing for Quiet Meadows", "b.pdf": "Harbor Security Notes"})
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(), "llama3", "Summarise the harbor paper",
                         [], lambda v, n: corpus.search(v, n), corpus=corpus)
    assert {c["source"] for c in p.chunks} == {"b.pdf"} and not p.broad


def test_breadth_questions_spread_over_sections(ollama):
    rows = [(f"{s}. Section {s}", f"Timers are discussed here, part {s}-{i}.") for s in range(1, 6) for i in range(4)]
    corpus = Corpus({"book.pdf": rows})
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(keyword_search=False), "llama3",
                         "Which chapters discuss timers?", [], lambda v, n: corpus.search(v, n), corpus=corpus)
    headings = [c["heading"] for c in p.chunks]
    assert len(p.chunks) == Settings().top_k + pipeline.BREADTH_EXTRA
    assert max(headings.count(h) for h in headings) <= pipeline.PER_SECTION
    assert len(set(headings)) >= 4
    assert "say which sections cover it" in p.prompt


def test_depth_questions_read_on_within_the_section(ollama):
    corpus = Corpus({"a.pdf": [("2. Method", "Step one sets the gain."), ("2. Method", "Step two sets the offset."),
                               ("2. Method", "Step three checks the relay."), ("2. Method", "Step four logs it."),
                               ("3. Results", "Results: all relays calm.")],
                     "b.pdf": [("", "Other.")]})
    corpus.search = lambda vector, n, key=None: [corpus._row("a.pdf", 0, 0.9)]
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(keyword_search=False), "llama3",
                         "Explain how the relay calibration is done", [],
                         lambda v, n: corpus.search(v, n), corpus=corpus)
    text = p.chunks[0]["text"]
    assert "Step two" in text and "Step three" in text and "Step four" in text
    assert "Results" not in text
    plain = pipeline.prepare(OllamaClient(ollama.url), Settings(keyword_search=False), "llama3",
                             "What gain is used?", [], lambda v, n: corpus.search(v, n), corpus=corpus)
    assert "Step three" not in plain.chunks[0]["text"]


def test_followups_also_search_with_the_previous_answer(ollama):
    corpus = Corpus({"a.pdf": [("", "Unrelated."), ("", "The second step sets the offset.")],
                     "b.pdf": [("", "Other.")]})
    searched = []

    def search(vector, n):
        searched.append(vector)
        if len(searched) == 3:              # the search made with the previous answer
            return [corpus._row("a.pdf", 1, 0.7)]
        return [corpus._row("a.pdf", 0, 0.8)]

    ollama.answers = ["Explain the second calibration step in more detail"]
    history = [{"q": "How are relays calibrated?", "a": "1. Set the gain. 2. Set the offset."}]
    p = pipeline.prepare(OllamaClient(ollama.url), Settings(keyword_search=False), "llama3",
                         "Explain the second step in more detail", history, search)
    assert len(searched) == 3
    assert any("second step sets the offset" in c["text"] for c in p.chunks)
