"""
Corpus BLEU (BLEU-4) in pure Python -- no nltk / sacrebleu dependency.

BLEU = BP * exp( mean_n log p_n ), where p_n is the modified (clipped) n-gram
precision and BP is the brevity penalty. We use a single reference per hypothesis
and a tiny floor on zero precisions (a simple smoothing) so a few missing n-grams
don't collapse the whole score to 0.

Inputs are TOKEN LISTS (already tokenised), e.g. ["thank", "you", "."].
"""

import math
from collections import Counter


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    """hyps, refs: lists of token-lists (one reference each). Returns BLEU in [0,1]."""
    assert len(hyps) == len(refs)
    clipped = [0] * (max_n + 1)
    total = [0] * (max_n + 1)
    hyp_len = ref_len = 0

    for hyp, ref in zip(hyps, refs):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            h = _ngrams(hyp, n)
            r = _ngrams(ref, n)
            total[n] += max(len(hyp) - n + 1, 0)
            for g, c in h.items():
                clipped[n] += min(c, r.get(g, 0))

    # modified precisions, averaged over only the n-gram orders that exist
    # ("effective order" smoothing), with a small floor so a single missing
    # n-gram doesn't zero the whole score.
    logs, orders = [], 0
    for n in range(1, max_n + 1):
        if total[n] == 0:
            continue                         # hypotheses too short for this order
        p = clipped[n] / total[n]
        logs.append(math.log(p if p > 0 else 1e-9))
        orders += 1
    geo = math.exp(sum(logs) / orders) if orders else 0.0

    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * geo


def sentence_bleu(hyp, ref, max_n=4):
    return corpus_bleu([hyp], [ref], max_n)


if __name__ == "__main__":
    # quick sanity checks
    a = "i declare the session of the parliament resumed .".split()
    partial = "i declare the parliament resumed today .".split()
    unrelated = "the cat sat on the mat".split()
    b_id = corpus_bleu([a], [a])
    b_part = corpus_bleu([partial], [a])
    b_bad = corpus_bleu([unrelated], [a])
    print(f"identical BLEU = {b_id:.3f}")
    print(f"partial   BLEU = {b_part:.3f}")
    print(f"unrelated BLEU = {b_bad:.3f}")
    assert abs(b_id - 1.0) < 1e-6, "identical should be 1.0"
    assert b_bad < b_part < b_id, "ordering check"
    print("bleu.py: sanity checks PASS")
