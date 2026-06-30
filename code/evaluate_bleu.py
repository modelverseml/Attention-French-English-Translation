"""
Compare the trained Transformer models on the test set by corpus BLEU.

For each model present, greedily translate the test French sentences and score
against the English references with bleu.corpus_bleu. Prints a ranked table.

    python evaluate_bleu.py            # default: first 300 test sentences
    python evaluate_bleu.py --n 1000
"""

import argparse
import time

import mt_data
from bleu import corpus_bleu
from translator import load_models

PAD, SOS, EOS = mt_data.PAD_ID, mt_data.SOS_ID, mt_data.EOS_ID


def ref_tokens(tgt_row, en_inv):
    out = []
    for i in tgt_row:
        i = int(i)
        if i == EOS:
            break
        if i in (PAD, SOS):
            continue
        out.append(en_inv.get(i, "<unk>"))
    return out


def score_models(models=None, n=300):
    """Return {model_name: BLEU in [0,1]} over the first n test sentences.
    Reusable by the CLI and the backend's /bleu endpoint."""
    fr_vocab, _, en_inv = mt_data.load_vocabs()
    src, tgt = mt_data.load_split("test")
    src, tgt = src[:n], tgt[:n]
    refs = [ref_tokens(row, en_inv) for row in tgt]
    fr_inv = {i: w for w, i in fr_vocab.items()}
    sources = [" ".join(fr_inv.get(int(i), "<unk>") for i in row
                        if int(i) not in (PAD, SOS, EOS)) for row in src]

    models = models if models is not None else load_models()
    scores = {}
    for name, fn in models.items():
        hyps = [fn(s).split() for s in sources]
        scores[name] = corpus_bleu(hyps, refs)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="how many test sentences to score")
    args = ap.parse_args()

    models = load_models()
    if not models:
        print("No trained models found — train at least one first "
              "(e.g. python seq2seq_pytorch.py).")
        return

    print(f"scoring {args.n} test sentences ...\n")
    t0 = time.time()
    scores = score_models(models, n=args.n)
    print("=== ranking ===")
    for name, b in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:11} {b * 100:5.2f}")
    print(f"\n({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
