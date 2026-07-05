# Attention (seq2seq) from Scratch — Derivation & Implementation

A sequence-to-sequence **French → English translator** built around an **attention
mechanism**, with the model itself written **from scratch in NumPy** (no
deep-learning framework). This repository has two parts:

1. **The theory** — a complete, hand-derived account of how additive (Bahdanau)
   attention works inside a GRU encoder-decoder: the alignment scores, the softmax
   weights, the context vector, and full **Backpropagation Through Time (BPTT)**
   through the attention block and both GRUs.
2. **A full-stack translator** — the same model implemented **three ways on
   identical data** (from scratch in NumPy, PyTorch, TensorFlow), trained on
   Europarl, compared with a from-scratch **BLEU** score, and served behind a
   **FastAPI + React** UI and a Streamlit app.

> Educational project: the goal is to make attention explicit and readable, not to
> be fast or state-of-the-art. Every backward equation in Part 1 matches
> [`code/seq2seq_manual.py`](code/seq2seq_manual.py) line for line — and that code
> passes a numerical gradient check.

---

# Part 1 — How Attention Works (Derivation)

## Table of Contents

- [1. The problem attention solves](#1-the-problem-attention-solves)
- [2. Notation & the model](#2-notation--the-model)
- [3. Forward pass](#3-forward-pass)
  - [3.1 Encoder (GRU)](#31-encoder-gru)
  - [3.2 Additive attention](#32-additive-attention)
  - [3.3 Decoder + output](#33-decoder--output)
- [4. Loss](#4-loss)
- [5. Backpropagation](#5-backpropagation)
  - [5.1 Output layer](#51-output-layer)
  - [5.2 Through the attention block](#52-through-the-attention-block)
  - [5.3 Into the encoder](#53-into-the-encoder)
- [6. Summary of attention gradients](#6-summary-of-attention-gradients)

> **Convention.** Batch-first / row-vector layout, matching the code: a linear map
> is `y = x·Wᵀ + b`, hidden states are rows, and element-wise product is `∘`.

---

## 1. The problem attention solves

A plain encoder-decoder compresses the **entire** source sentence into one fixed
vector (the encoder's last state) and asks the decoder to generate the whole target
from it. For long sentences that vector is an information bottleneck.

**Attention removes the bottleneck.** At every decoding step the decoder looks back
at *all* encoder states and takes a **weighted average** of them, with the weights
computed on the fly from how relevant each source position is to what it's about to
generate. Nothing is squeezed through a single vector.

---

## 2. Notation & the model

| Symbol | Meaning |
|---|---|
| `x₁..x_S`, `y₁..y_T` | source / target token ids |
| `H` | hidden size |
| `hᵢ ∈ ℝᴴ` | encoder state at source position `i` |
| `sₜ ∈ ℝᴴ` | decoder state at target step `t` |
| `αₜ,ᵢ` | attention weight (step `t` over source `i`) |
| `cₜ ∈ ℝᴴ` | context vector at step `t` |

Parameters: source/target embeddings `Eˢ, Eᵗ`; encoder GRU; decoder GRU; attention
`W_s, W_h ∈ ℝᴴˣᴴ` and `v ∈ ℝᴴ`; output `W_y ∈ ℝ^{V×2H}, b_y`.

---

## 3. Forward pass

### 3.1 Encoder (GRU)

Embed the source, then run a GRU over it to get one state per source token:

$$h_i = \mathrm{GRU_{enc}}(h_{i-1},\, E^s[x_i]), \qquad i = 1..S,\quad h_0 = 0$$

where a GRU cell (update gate `z`, reset gate `r`, candidate `h̃`) is

$$z = \sigma(W_z[h_{i-1},x]),\quad r = \sigma(W_r[h_{i-1},x]),$$
$$\tilde h = \tanh(W_h[\,r \circ h_{i-1},\, x\,]),\qquad h_i = (1-z)\circ h_{i-1} + z\circ \tilde h .$$

(The full GRU gate BPTT is derived in the sibling **GRU** project; here we focus on
the attention block and the seq2seq wiring.) The decoder state is initialised
`s₀ = 0`, so the model relies on attention to read the source.

### 3.2 Additive attention

This is the heart of the model. At decoder step `t`, using the previous decoder
state `s_{t-1}` and **every** encoder state `hᵢ`:

**Alignment score** — an additive (Bahdanau) score of "how relevant is source `i`
to what I'm about to decode":

$$u_{t,i} = \tanh(W_s h_i + W_h s_{t-1}) \in \mathbb{R}^H, \qquad e_{t,i} = v^\top u_{t,i} \in \mathbb{R}.$$

**Weights** — softmax over source positions (padding positions set to `−∞`, so they
get weight 0):

$$\boxed{\;\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{S}\exp(e_{t,j})}\;}$$

**Context** — the weighted average of encoder states:

$$\boxed{\; c_t = \sum_{i=1}^{S} \alpha_{t,i}\, h_i \;}$$

`W_s hᵢ` (the "keys") is the same at every step, so it's computed once and reused.

### 3.3 Decoder + output

Feed the previous target word's embedding **and** the context into the decoder GRU,
then project the decoder state together with the context to vocabulary logits:

$$s_t = \mathrm{GRU_{dec}}\big(s_{t-1},\, [\,E^t[y_{t-1}],\, c_t\,]\big),$$
$$\text{logits}_t = [\,s_t,\, c_t\,]\,W_y^\top + b_y, \qquad \hat y_t = \mathrm{softmax}(\text{logits}_t).$$

Training uses **teacher forcing** (`y_{t-1}` is the ground-truth previous token);
inference feeds back the model's own greedy prediction.

---

## 4. Loss

Cross-entropy over the non-padding target tokens, averaged by the token count `N`:

$$L = -\frac{1}{N}\sum_{t:\,y_t \neq \text{pad}} \log \hat y_t[y_t].$$

---

## 5. Backpropagation

The backward pass walks the graph in reverse:
**output → decoder GRU (BPTT) → attention → encoder GRU (BPTT) → embeddings.** The
softmax+cross-entropy gradient (derived in the RNN project) is reused per step:

$$\frac{\partial L}{\partial \text{logits}_t} = \big(\hat y_t - \mathbf{1}[y_t]\big)\cdot \frac{\text{mask}_t}{N}.$$

### 5.1 Output layer

With `oₜ = [sₜ, cₜ]`:

$$\frac{\partial L}{\partial W_y} = \sum_t \Big(\tfrac{\partial L}{\partial \text{logits}_t}\Big)^\top o_t,\qquad \frac{\partial L}{\partial b_y}=\sum_t \tfrac{\partial L}{\partial \text{logits}_t},$$
$$do_t = \tfrac{\partial L}{\partial \text{logits}_t}\,W_y \;\Rightarrow\; ds_t^{\text{(out)}} = do_t[:H],\quad dc_t^{\text{(out)}} = do_t[H{:}].$$

The decoder GRU backward (standard GRU BPTT) takes the total `dsₜ = ds_t^{(out)} +
ds_t^{(future)}` and returns: the decoder gate-weight gradients, the gradient into
its input `[Eᵗ[y_{t-1}], cₜ]` — which splits into an embedding gradient and a second
context gradient `dc_t^{(gru)}` — and the gradient into `s_{t-1}`.

Total context gradient: `dcₜ = dc_t^{(out)} + dc_t^{(gru)}`.

### 5.2 Through the attention block

This is the part unique to attention. Start from `dcₜ` and `cₜ = Σᵢ αₜ,ᵢ hᵢ`.

**Into the weights and the encoder states:**

$$d\alpha_{t,i} = dc_t \cdot h_i \quad(\text{dot over } H), \qquad dh_i \mathrel{+}= \alpha_{t,i}\, dc_t.$$

**Through the softmax** (`αₜ = softmax(eₜ)`), the standard Jacobian gives

$$\boxed{\; de_{t,i} = \alpha_{t,i}\Big(d\alpha_{t,i} - \sum_{j}\alpha_{t,j}\,d\alpha_{t,j}\Big) \;}$$

(and `deₜ,ᵢ = 0` at padded positions, since `αₜ,ᵢ = 0` there).

**Through the score** `eₜ,ᵢ = vᵀ uₜ,ᵢ` and the `tanh`:

$$dv \mathrel{+}= \sum_i de_{t,i}\, u_{t,i}, \qquad du_{t,i} = de_{t,i}\, v, \qquad dp_{t,i} = du_{t,i}\circ\big(1 - u_{t,i}^2\big).$$

**Into the two projections** (`uₜ,ᵢ = tanh(hᵢWₛᵀ + s_{t-1}Wₕᵀ)`), accumulating over
source positions:

$$\frac{\partial L}{\partial W_s} \mathrel{+}= \sum_i dp_{t,i}^\top h_i, \qquad dh_i \mathrel{+}= dp_{t,i}\,W_s,$$
$$\frac{\partial L}{\partial W_h} \mathrel{+}= \Big(\textstyle\sum_i dp_{t,i}\Big)^\top s_{t-1}, \qquad ds_{t-1} \mathrel{+}= \Big(\textstyle\sum_i dp_{t,i}\Big) W_h.$$

The attention gradient into `s_{t-1}` is **added to** the decoder-GRU gradient into
`s_{t-1}` (the same state is used both as the GRU's previous state and as the
attention query), and then flows one more step back through the decoder.

### 5.3 Into the encoder

Every decoder step contributes to every `dhᵢ` (through the context **and** through
the key projection `Wₛhᵢ`). After all `T` decoder steps, the accumulated `dhᵢ` is
backpropagated through the encoder GRU (BPTT over the source), which finally yields
the source-embedding gradients. Target-embedding gradients are scattered from the
`demb_t` produced in §5.1.

---

## 6. Summary of attention gradients

| Quantity | Gradient |
|---|---|
| context | `cₜ = Σᵢ αₜ,ᵢ hᵢ` |
| weight `∂L/∂αₜ,ᵢ` | `dcₜ · hᵢ` |
| encoder state (from context) | `dhᵢ += αₜ,ᵢ dcₜ` |
| softmax `∂L/∂eₜ,ᵢ` | `αₜ,ᵢ(dαₜ,ᵢ − Σⱼ αₜ,ⱼ dαₜ,ⱼ)` |
| `∂L/∂v` | `Σᵢ deₜ,ᵢ uₜ,ᵢ` |
| pre-tanh `dpₜ,ᵢ` | `(deₜ,ᵢ v) ∘ (1 − uₜ,ᵢ²)` |
| `∂L/∂Wₛ` | `Σᵢ dpₜ,ᵢᵀ hᵢ`;  `dhᵢ += dpₜ,ᵢ Wₛ` |
| `∂L/∂Wₕ` | `(Σᵢ dpₜ,ᵢ)ᵀ s_{t-1}`;  `ds_{t-1} += (Σᵢ dpₜ,ᵢ) Wₕ` |

`seq2seq_manual.py` implements exactly these and passes a directional gradient check
(analytic vs. finite difference over all parameters) at ~1e-11.

---

# Part 2 — The Translator (Code & App)

Three implementations on identical data, so they can be compared:

| Implementation | File | Notes |
|---|---|---|
| **Manual (NumPy)** | `code/seq2seq_manual.py` | the Part 1 derivation, in code — gradient-checked |
| **PyTorch** | `code/seq2seq_pytorch.py` | `nn.GRU` enc/dec + Bahdanau attention |
| **TensorFlow** | `code/seq2seq_tensorflow.py` | same architecture in Keras, custom train loop |

Shared architecture: **GRU encoder → Bahdanau attention → GRU decoder**, a learned
embedding over a capped vocabulary, teacher forcing for training, greedy decoding.

## Data

Source corpus: **Europarl v7 fr-en** (`fr-en/`, ~2M pairs, 618 MB) — too large to
train on a laptop. `data_prep.py` streams it into a small, length-capped subset:

- pairs with ≤ 12 tokens on both sides · 50k train / 2k dev / 2k test
- vocabulary capped at 8k words **per language** (+ `<pad> <sos> <eos> <unk>`)
- saved as padded integer-id arrays under `data/processed/`

```bash
cd code
python data_prep.py                 # writes data/processed/{train,dev,test}.npz + vocabs
python data_prep.py --max_len 15 --n_train 100000 --vocab_size 12000   # tune
```

A trainable embedding over the capped vocab is the **best + lowest-storage** choice:
no multi-GB pretrained vectors, and unlike one-hot it doesn't blow the input
dimension up to the vocabulary size.

## Train + translate

```bash
cd code
python seq2seq_pytorch.py           # ~minutes on CPU
python seq2seq_tensorflow.py
python seq2seq_manual.py            # runs the gradient check; 'python seq2seq_manual.py train' to train + save
```

## Compare models (BLEU)

`evaluate_bleu.py` translates the test set with every trained model and scores it
with a from-scratch corpus **BLEU-4** (`bleu.py`, no nltk/sacrebleu):

```bash
cd code && python evaluate_bleu.py --n 500
```

## Translation UI

**React + FastAPI** (two terminals):
```bash
cd code/backend && pip install -r requirements.txt && uvicorn app:app --reload --port 8000
cd frontend && npm install && npm run dev          # http://localhost:5173
```
**Streamlit**: `streamlit run streamlit_app.py`

Both reuse `code/translator.py`, so they show whatever models are trained (and a
per-model BLEU badge).

## Layout

```
Attention/
├── fr-en/                       # raw Europarl corpus (gitignored — large)
├── data/processed/              # subset: padded id arrays + vocab json + config
├── data/model_artifacts/        # saved trained models
├── streamlit_app.py             # translation UI
└── code/
    ├── data_prep.py             # corpus -> small processed subset + vocab
    ├── mt_data.py               # shared loading / encode / decode
    ├── seq2seq_pytorch.py
    ├── seq2seq_tensorflow.py
    ├── seq2seq_manual.py        # from scratch + gradient check
    ├── translator.py            # loads trained models, translate(sentence)
    ├── bleu.py                  # corpus BLEU-4
    ├── evaluate_bleu.py         # BLEU comparison table
    └── backend/                 # FastAPI (predictor + app)
```

## See also

A **Transformer** translator (self-attention, no recurrence) lives in the sibling
`Transformer/` project — same data pipeline, same UI + BLEU tooling, with its own
from-scratch derivation.
