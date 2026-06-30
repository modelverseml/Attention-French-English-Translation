"""
Load whatever trained seq2seq+attention models exist and translate French ->
English with each. Used by the Streamlit UI and the BLEU evaluation. Models whose
artifact hasn't been trained yet are skipped.

    pytorch    -> data/model_artifacts/pytorch_seq2seq.pt
    tensorflow -> data/model_artifacts/tensorflow_seq2seq.weights.h5
    manual     -> data/model_artifacts/manual_seq2seq.npz
"""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import mt_data

ARTIFACTS = Path(__file__).resolve().parent.parent / "data" / "model_artifacts"
FRAMEWORKS = ["pytorch", "tensorflow", "manual"]


def load_models(which=None):
    cfg = mt_data.load_config()
    fr_vocab, _, en_inv = mt_data.load_vocabs()
    want = which or FRAMEWORKS
    models = {}

    if "pytorch" in want and (ARTIFACTS / "pytorch_seq2seq.pt").exists():
        import torch
        import seq2seq_pytorch as PT
        m = PT.Seq2Seq(cfg["vocab_src"], cfg["vocab_tgt"])
        m.load_state_dict(torch.load(ARTIFACTS / "pytorch_seq2seq.pt", map_location="cpu"))
        m.eval()
        models["pytorch"] = lambda s, m=m: PT.translate(m, s, fr_vocab, en_inv, cfg)

    if "tensorflow" in want and (ARTIFACTS / "tensorflow_seq2seq.weights.h5").exists():
        import seq2seq_tensorflow as TF
        m = TF.Seq2Seq(cfg["vocab_src"], cfg["vocab_tgt"])
        m(np.zeros((1, cfg["src_len"]), np.int32), np.zeros((1, cfg["tgt_len"]), np.int32))
        m.load_weights(str(ARTIFACTS / "tensorflow_seq2seq.weights.h5"))
        models["tensorflow"] = lambda s, m=m: TF.translate(m, s, fr_vocab, en_inv, cfg)

    if "manual" in want and (ARTIFACTS / "manual_seq2seq.npz").exists():
        import seq2seq_manual as MN
        m = MN.ManualSeq2SeqAttention.load(ARTIFACTS / "manual_seq2seq.npz")

        def _manual(s, m=m):
            ids = m.translate_ids(mt_data.encode_source(s, fr_vocab, cfg["src_len"]))
            return mt_data.decode_target(ids, en_inv)

        models["manual"] = _manual

    return models


def translate_all(sentence, models=None):
    models = models or load_models()
    return {name: fn(sentence) for name, fn in models.items()}


if __name__ == "__main__":
    models = load_models()
    if not models:
        print("No trained models found. Train at least one, e.g.:\n"
              "  python seq2seq_pytorch.py\n  python seq2seq_manual.py train")
    else:
        print("loaded:", list(models))
        for s in ["je vous remercie .", "merci beaucoup .", "le parlement européen"]:
            print(f"\nfr: {s}")
            for name, en in translate_all(s, models).items():
                print(f"  {name:11} -> {en}")
