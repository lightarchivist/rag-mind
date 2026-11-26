"""Keyword search (BM25), used alongside similarity search.

Similarity search finds passages about the same topic, but it is weak at exact
terms such as "label 3", "UDP 646" or "Adj-SID". In a collection of books on
related subjects, a passage from a neighbouring chapter can be more similar
than the one that holds the answer. Keyword search ranks the passages that
contain the question's distinctive words; pipeline.fuse merges both rankings.
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


def tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if re.search(r"[a-z]", tok) and _PARTS.search(tok):
            out.extend(p for p in _PARTS.split(tok) if p)     # "ti-lfa" also matches "lfa"
    return [t for t in out if t not in STOPWORDS]


class KeywordIndex:
    """BM25 over a list of passages; results are positions in that list."""

    def __init__(self, texts: list[str]):
        self.size = len(texts)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, text in enumerate(texts):
            toks = tokens(text)
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.average = (sum(self.lengths) / self.size) if self.size else 1.0

    def search(self, query: str, n: int, allowed=None) -> list[tuple[int, float]]:
        """Best `n` passages as (position, score). A passage must contain at least
        two of the query's distinctive words (one, if the query has only one), so a
        single common word never brings in an unrelated passage."""
        if not self.size or n < 1:
            return []
        terms = [t for t in dict.fromkeys(tokens(query)) if t in self.postings]
        distinctive = {t for t in terms if len(self.postings[t]) <= max(5, COMMON * self.size)}
        if not distinctive:
            return []
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, int] = defaultdict(int)
        for term in terms:
            posting = self.postings[term]
            idf = math.log(1 + (self.size - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting:
                if allowed is not None and i not in allowed:
                    continue
                norm = tf + K1 * (1 - B + B * self.lengths[i] / self.average)
                scores[i] += idf * tf * (K1 + 1) / norm
                if term in distinctive:
                    matched[i] += 1
        need = min(2, len(distinctive))
        hits = [(i, s) for i, s in scores.items() if matched[i] >= need]
        return sorted(hits, key=lambda h: -h[1])[:n]
