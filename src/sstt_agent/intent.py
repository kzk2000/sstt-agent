import math
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")


def tokenize_words(text: str):
    return WORD_RE.findall(text)


def work_cost(word: str) -> int:
    cost = len(word)
    if word and word[0].isupper():
        cost += 1
    return cost


def compute_token_intent(text: str, coherence: int = 45, rho: float = 0.1):
    tokens = tokenize_words(text)
    freq = Counter(tokens)
    num_sents = max(1, len(re.findall(r"[.!?]", text)) or 1)
    Phi0 = 1 / coherence
    scores = {}
    for w, f in freq.items():
        W = work_cost(w)
        Phi = f / num_sents
        scores[w] = (Phi * W) / (1 + math.exp(Phi / Phi0 - rho))
    return scores


def phrase_intent(phrase: str, token_scores: dict) -> float:
    toks = tokenize_words(phrase)
    if not toks:
        return 0.0
    vals = []
    for t in toks:
        vals.append(token_scores.get(t, token_scores.get(t.capitalize(), 0.0)))
    return max(vals) if vals else 0.0
