# French → English Translation with Attention

A small **sequence-to-sequence translator** (French → English) built around an
**attention mechanism**, implemented **three ways on identical data** so they can be
compared:

| Implementation | File | Notes |
|---|---|---|
| **Manual (NumPy)** | `code/seq2seq_manual.py` | from scratch — forward + full BPTT through embeddings, encoder GRU, **additive attention**, decoder GRU, softmax. **Gradient-checked.** |
| **PyTorch** | `code/seq2seq_pytorch.py` | `nn.GRU` enc/dec + Bahdanau attention |
| **TensorFlow** | `code/seq2seq_tensorflow.py` | same architecture in Keras, custom train loop |

All three share one architecture: **GRU encoder → Bahdanau (additive) attention →
GRU decoder**, with a learned embedding layer over a capped vocabulary, teacher
forcing for training, and greedy decoding for translation.

## Data

Source corpus: **Europarl v7 fr-en** (`fr-en/`, ~2M sentence pairs, 618 MB) — far
too large to train on a laptop. `data_prep.py` streams it and builds a small,
length-capped subset:

- keep pairs with ≤ 12 tokens on both sides
- 50k train / 2k dev / 2k test
- vocabulary capped at 8k words **per language** (+ `<pad> <sos> <eos> <unk>`)
- everything saved as padded integer-id arrays under `data/processed/`

```bash
cd code
python data_prep.py                 # writes data/processed/{train,dev,test}.npz + vocabs
# tune the subset:
python data_prep.py --max_len 15 --n_train 100000 --vocab_size 12000
```

### Why a learned embedding (not pretrained / one-hot)?

A trainable embedding layer over the capped vocab is the **best + lowest-storage**
choice: no multi-GB pretrained vectors to ship, and unlike one-hot it doesn't blow
the input dimension up to the vocabulary size. It's trained jointly with the model.

## Train + translate

Each implementation is runnable on its own (trains, then prints sample greedy
translations):

```bash
cd code
python seq2seq_pytorch.py           # ~minutes on CPU
python seq2seq_tensorflow.py
python seq2seq_manual.py            # runs the gradient check; import to train (slowest)
```

`seq2seq_manual.py` run on its own performs a **gradient check** (analytic vs.
finite-difference, directional) — it should print `PASS` with rel-err ~1e-11.

## Status / results

Smoke-trained on small slices (a few epochs) to validate correctness:

- **PyTorch** — `je vous remercie .` → `thank you .`
- **TensorFlow** — learning (loss drops steadily)
- **Manual NumPy** — gradient check **PASS**; `merci beaucoup .` → `thank you , commissioner .`

Quality improves with full training (50k × 10 epochs); the manual version is the
slowest (pure-Python BPTT, ~30+ min) but produces real translations.

## Compare models (BLEU)

`evaluate_bleu.py` greedily translates the test set with every trained model and
scores it against the references using a from-scratch corpus **BLEU-4** (`bleu.py`,
no nltk/sacrebleu). It prints a ranked table; models you haven't trained are skipped.

```bash
cd code
python evaluate_bleu.py --n 500
# e.g.  pytorch  BLEU= 9.24   (rises a lot with full training)
```

## Translation UI

Type French, see each trained model's English side by side. Two options:

**React + FastAPI** (two terminals):
```bash
cd code/backend && pip install -r requirements.txt && uvicorn app:app --reload --port 8000
cd frontend && npm install && npm run dev          # http://localhost:5173
```
The Vite dev server proxies `/api` → the backend on `:8000` (start the backend first).

**Streamlit** (single command):
```bash
streamlit run streamlit_app.py
```

Both reuse `code/translator.py`, so they show whatever models are trained.

## Layout

```
Attenuation/
├── fr-en/                       # raw Europarl corpus (gitignored — large)
├── data/processed/              # subset: padded id arrays + vocab json + config
├── data/model_artifacts/        # saved trained models
├── streamlit_app.py             # translation UI
└── code/
    ├── data_prep.py             # corpus -> small processed subset + vocab
    ├── mt_data.py               # shared loading / encode / decode
    ├── seq2seq_pytorch.py        # (run to train -> pytorch_seq2seq.pt)
    ├── seq2seq_tensorflow.py
    ├── seq2seq_manual.py         # from scratch + gradient check ('… train' to save)
    ├── translator.py             # loads trained models, translate(sentence)
    ├── bleu.py                   # corpus BLEU-4
    └── evaluate_bleu.py          # BLEU comparison table
```

## See also

A **Transformer** translator (self-attention, no recurrence) lives in the sibling
`Transformer/` project — same data pipeline, same UI + BLEU tooling.
