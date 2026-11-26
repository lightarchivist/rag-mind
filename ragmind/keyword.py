"""Keyword search (BM25), used alongside similarity search.

Similarity search finds passages about the same topic, but it is weak at exact
terms such as "label 3", "UDP 646" or "Adj-SID". In a collection of books on
related subjects, a passage from a neighbouring chapter can be more similar
than the one that holds the answer. Keyword search ranks the passages that
contain the question's distinctive words and phrases; pipeline.fuse adds
the best of them to the similarity results.
"""
import math
import re
from collections import Counter, defaultdict

# Words, plus technical tokens kept whole: "ti-lfa", "r5.2", "802.1qbv", "10.1.1.1".
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
_PARTS = re.compile(r"[-./]")
STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers him his how i if in into is it its itself just me more
most my no nor not now of off on once only or other our out over own same she should so some
such than that the their them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours
explain describe tell show give list please according book document documents paper papers
""".split())
K1, B = 1.2, 0.75
COMMON = 0.05           # a word in more than 5% of passages doesn't count as distinctive


def stem(word: str) -> str:
    """A light stemmer, so "chunks", "chunked" and "chunking" are one term.
    Only plain words are changed; "ti-lfa", "802.1q" and "ospfv3" stay as they are.
    It needs to be consistent rather than linguistically right: the same rules
    run over the passages and the question."""
    if not word.isalpha() or len(word) <= 3:
        return word
    w = word
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
        w = w[:-1] if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz" else w
    elif w.endswith("ed") and len(w) > 5:
        w = w[:-2]
        w = w[:-1] if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz" else w
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def _words(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _parts(word: str) -> list[str]:
    if re.search(r"[a-z]", word) and _PARTS.search(word):
        return [p for p in _PARTS.split(word) if p]           # "ti-lfa" also matches "lfa"
    return []


def terms(text: str) -> list[str]:
    """Index terms: words, the parts of hyphenated words, and two-word phrases,
    so "label 3", "udp 646" and "type 2" are terms of their own."""
    words = _words(text)
    out = []
    for w in words:
        out.append(w)
        out.extend(_parts(w))
    out.extend(f"{a} {b}" for a, b in zip(words, words[1:]))
    return out


def tokens(text: str) -> list[str]:
    """Single-word terms only (used in tests and for display)."""
    return [t for t in terms(text) if " " not in t]


def query_groups(text: str) -> list[set[str]]:
    """The query as groups of equivalent terms: a word with its parts, or a phrase.
    A passage matches a group if it contains any of its terms."""
    words = _words(text)
    groups = [{w, *_parts(w)} for w in dict.fromkeys(words)]
    groups += [{f"{a} {b}"} for a, b in dict.fromkeys(zip(words, words[1:]))]
    return groups


class KeywordIndex:
    """BM25 over a list of passages; results are positions in that list."""

    def __init__(self, texts: list[str]):
        self.size = len(texts)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, text in enumerate(texts):
            toks = terms(text)
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.average = (sum(self.lengths) / self.size) if self.size else 1.0

    def _distinctive(self, term: str) -> bool:
        return 0 < len(self.postings.get(term, ())) <= max(5, COMMON * self.size)

    def search(self, query: str, n: int, allowed=None) -> list[tuple[int, float]]:
        """Best `n` passages as (position, score). A passage must match at least two
        distinctive parts of the query (words or phrases), so one word that happens
        to be rare, such as "reserved", never brings in an unrelated passage."""
        if not self.size or n < 1:
            return []
        groups = [g for g in query_groups(query) if any(self._distinctive(t) for t in g)]
        if len(groups) < 2:
            return []
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, set] = defaultdict(set)
        for gi, group in enumerate(groups):
            for term in group:
                posting = self.postings.get(term)
                if not posting:
                    continue
                idf = math.log(1 + (self.size - len(posting) + 0.5) / (len(posting) + 0.5))
                for i, tf in posting:
                    if allowed is not None and i not in allowed:
                        continue
                    norm = tf + K1 * (1 - B + B * self.lengths[i] / self.average)
                    scores[i] += idf * tf * (K1 + 1) / norm
                    matched[i].add(gi)
        hits = [(i, s) for i, s in scores.items() if len(matched[i]) >= 2]
        return sorted(hits, key=lambda h: -h[1])[:n]
