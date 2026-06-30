"""
Prepare the Europarl fr->en corpus for a small attention-based translator.

The raw corpus is ~2M long sentence pairs (>600 MB) -- far too much to train on
a laptop. So we STREAM the files, keep only short pairs (<= MAX_LEN tokens on
both sides), take a small subset, build a capped vocabulary per language, encode
everything to padded integer id arrays, and save:

    data/processed/vocab_fr.json   word -> id   (source)
    data/processed/vocab_en.json   word -> id   (target)
    data/processed/{train,dev,test}.npz   src (B, S) , tgt (B, T)   int32

Source sequences are  [tokens..., <eos>]            padded with <pad>.
Target sequences are  [<sos>, tokens..., <eos>]     padded with <pad>.

The same arrays/vocab are used by all three implementations (PyTorch,
TensorFlow, manual NumPy), so they train and translate on identical data.
"""

import json
import re
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "fr-en"
SRC_FILE = RAW_DIR / "europarl-v7.fr-en.fr"   # source = French
TGT_FILE = RAW_DIR / "europarl-v7.fr-en.en"   # target = English
OUT_DIR = REPO_ROOT / "data" / "processed"

# special tokens (ids 0..3 reserved)
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# defaults (tunable from the CLI)
MAX_LEN = 12          # keep pairs with <= this many tokens on BOTH sides
VOCAB_SIZE = 8000     # top-k words per language (plus the 4 specials)
N_TRAIN = 50000
N_DEV = 2000
N_TEST = 2000
MAX_SCAN = 600000     # how many raw lines to read while collecting short pairs

_token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    # lowercase, then split into word / punctuation tokens (accent-aware)
    return _token_re.findall(text.strip().lower())


def collect_pairs(max_len, n_total, max_scan):
    # stream both files in parallel, keep short pairs until we have enough
    pairs = []
    with open(SRC_FILE, encoding="utf-8") as fsrc, open(TGT_FILE, encoding="utf-8") as ftgt:
        for i, (s, t) in enumerate(zip(fsrc, ftgt)):
            if i >= max_scan or len(pairs) >= n_total:
                break
            st, tt = tokenize(s), tokenize(t)
            if 1 <= len(st) <= max_len and 1 <= len(tt) <= max_len:
                pairs.append((st, tt))
    return pairs


def build_vocab(token_lists, vocab_size):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for word, _ in counter.most_common(vocab_size):
        vocab[word] = len(vocab)
    return vocab


def encode(tokens, vocab, add_sos, add_eos, max_len):
    ids = [vocab.get(w, UNK_ID) for w in tokens]
    if add_sos:
        ids = [SOS_ID] + ids
    if add_eos:
        ids = ids + [EOS_ID]
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


def encode_split(pairs, vocab_src, vocab_tgt, max_len):
    s_len = max_len + 1   # + <eos>
    t_len = max_len + 2   # + <sos> + <eos>
    src = np.array([encode(s, vocab_src, False, True, s_len) for s, _ in pairs], dtype=np.int32)
    tgt = np.array([encode(t, vocab_tgt, True, True, t_len) for _, t in pairs], dtype=np.int32)
    return src, tgt


def parse_args():
    p = argparse.ArgumentParser(description="prepare the Europarl fr->en subset")
    p.add_argument("--max_len", type=int, default=MAX_LEN)
    p.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    p.add_argument("--n_train", type=int, default=N_TRAIN)
    p.add_argument("--n_dev", type=int, default=N_DEV)
    p.add_argument("--n_test", type=int, default=N_TEST)
    p.add_argument("--max_scan", type=int, default=MAX_SCAN)
    return p.parse_args()


def main():
    args = parse_args()
    n_total = args.n_train + args.n_dev + args.n_test
    print(f"scanning up to {args.max_scan} lines for <= {args.max_len}-token pairs ...")
    pairs = collect_pairs(args.max_len, n_total, args.max_scan)
    print(f"collected {len(pairs)} short pairs")
    if len(pairs) < n_total:
        print(f"  (wanted {n_total}; raise --max_len or --max_scan for more)")

    train = pairs[:args.n_train]
    dev = pairs[args.n_train:args.n_train + args.n_dev]
    test = pairs[args.n_train + args.n_dev:n_total]

    # vocab from TRAIN only, so dev/test can't leak rare words into the vocab
    vocab_src = build_vocab([s for s, _ in train], args.vocab_size)
    vocab_tgt = build_vocab([t for _, t in train], args.vocab_size)
    print(f"vocab: fr={len(vocab_src)}  en={len(vocab_tgt)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "vocab_fr.json").write_text(json.dumps(vocab_src, ensure_ascii=False))
    (OUT_DIR / "vocab_en.json").write_text(json.dumps(vocab_tgt, ensure_ascii=False))

    for name, split in (("train", train), ("dev", dev), ("test", test)):
        if not split:
            continue
        src, tgt = encode_split(split, vocab_src, vocab_tgt, args.max_len)
        np.savez_compressed(OUT_DIR / f"{name}.npz", src=src, tgt=tgt)
        print(f"saved {name}: src={src.shape} tgt={tgt.shape}")

    # also save the config so the models know the shapes / vocab sizes
    cfg = {
        "max_len": args.max_len,
        "src_len": args.max_len + 1,
        "tgt_len": args.max_len + 2,
        "vocab_src": len(vocab_src),
        "vocab_tgt": len(vocab_tgt),
        "pad_id": PAD_ID, "sos_id": SOS_ID, "eos_id": EOS_ID, "unk_id": UNK_ID,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2))
    print("config:", cfg)


if __name__ == "__main__":
    main()
