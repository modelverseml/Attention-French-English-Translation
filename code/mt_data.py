"""
Shared data access for the translator: load the processed splits, the vocab,
and helpers to encode a raw French sentence / decode English ids back to text.
Used by all three implementations so they share identical preprocessing.
"""

import json
from pathlib import Path

import numpy as np

from data_prep import tokenize, PAD_ID, SOS_ID, EOS_ID, UNK_ID

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_config():
    return json.loads((PROC / "config.json").read_text())


def load_vocabs():
    fr = json.loads((PROC / "vocab_fr.json").read_text())
    en = json.loads((PROC / "vocab_en.json").read_text())
    en_inv = {i: w for w, i in en.items()}
    return fr, en, en_inv


def load_split(name):
    npz = np.load(PROC / f"{name}.npz")
    return npz["src"], npz["tgt"]


def encode_source(sentence, fr_vocab, src_len):
    # raw French -> padded source id row (tokens + <eos>), matching data_prep
    ids = [fr_vocab.get(w, UNK_ID) for w in tokenize(sentence)]
    ids = ids[: src_len - 1] + [EOS_ID]
    ids += [PAD_ID] * (src_len - len(ids))
    return np.array(ids, dtype=np.int64)


def decode_target(ids, en_inv):
    # english ids -> string, stopping at <eos>, skipping specials
    words = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i in (PAD_ID, SOS_ID):
            continue
        words.append(en_inv.get(i, "<unk>"))
    return " ".join(words)
