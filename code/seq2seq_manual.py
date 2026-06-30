"""
French -> English translator: GRU encoder-decoder with Bahdanau attention,
written FROM SCRATCH in NumPy -- no deep-learning library. Same architecture as
seq2seq_pytorch.py / seq2seq_tensorflow.py.

This is the educational centerpiece: every gradient is hand-derived and the
backward pass walks the whole computation graph in reverse --

    output softmax  ->  decoder GRU (BPTT)  ->  additive attention
                    ->  encoder GRU (BPTT)  ->  source/target embeddings

Training uses full teacher forcing + Adam; translation uses greedy decoding.
`gradient_check()` verifies the analytic gradients against finite differences.

Conventions: batch-first, row vectors; a gate is `sigmoid(concat @ W.T + b)` with
weights stored as (out, in). Decoder hidden is initialised to zeros, so the model
relies on attention to read the source (a common, simple choice).
"""

from pathlib import Path

import numpy as np

import mt_data

ARTIFACTS = Path(__file__).resolve().parent.parent / "data" / "model_artifacts"
EMB_DIM = 96
HID = 128
EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-3
SEED = 42
NEG = -1e9   # masked-attention score


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class ManualSeq2SeqAttention:
    def __init__(self, vocab_src=None, vocab_tgt=None, emb=EMB_DIM, hid=HID, seed=SEED):
        if vocab_src is None:
            return  # empty shell for load()
        rng = np.random.default_rng(seed)
        E, H = emb, hid

        def gate(out_dim, in_dim):
            return rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim)

        self.E, self.H = E, H
        self.params = {
            "Es": rng.standard_normal((vocab_src, E)) * 0.1,
            "Et": rng.standard_normal((vocab_tgt, E)) * 0.1,
            # encoder GRU: concat = [h_prev(H), x(E)]
            "eWz": gate(H, H + E), "eWr": gate(H, H + E), "eWh": gate(H, H + E),
            "ebz": np.zeros((1, H)), "ebr": np.zeros((1, H)), "ebh": np.zeros((1, H)),
            # attention
            "aWs": gate(H, H), "aWh": gate(H, H), "av": gate(1, H).T,   # (H,1)
            # decoder GRU: concat = [s_prev(H), x(E+H)]
            "dWz": gate(H, H + E + H), "dWr": gate(H, H + E + H), "dWh": gate(H, H + E + H),
            "dbz": np.zeros((1, H)), "dbr": np.zeros((1, H)), "dbh": np.zeros((1, H)),
            # output over [s_t(H), context(H)]
            "Wy": gate(vocab_tgt, 2 * H), "by": np.zeros((1, vocab_tgt)),
        }
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # ------------------------- forward -------------------------
    def forward_loss(self, src, tgt):
        """Full teacher-forced forward. Returns (loss, cache)."""
        p = self.params
        E, H = self.E, self.H
        B, S = src.shape
        T = tgt.shape[1]

        # ---- encoder ----
        src_emb = p["Es"][src]                       # (B,S,E)
        enc_states = np.zeros((B, S, H))
        enc_cache = []
        h = np.zeros((B, H))
        for s in range(S):
            x = src_emb[:, s, :]
            concat = np.concatenate([h, x], axis=1)
            z = _sigmoid(concat @ p["eWz"].T + p["ebz"])
            r = _sigmoid(concat @ p["eWr"].T + p["ebr"])
            concat_r = np.concatenate([r * h, x], axis=1)
            hh = np.tanh(concat_r @ p["eWh"].T + p["ebh"])
            h = (1 - z) * h + z * hh
            enc_states[:, s, :] = h
            enc_cache.append((h.copy(), z, r, hh, x))  # note: store h_prev below
        # fix: we stored h_next; rebuild with h_prev for backward
        # (recompute prevs cheaply from enc_states)
        src_mask = src != mt_data.PAD_ID            # (B,S)

        proj_s = enc_states @ p["aWs"].T            # (B,S,H)

        # ---- decoder ----
        s_prev = np.zeros((B, H))
        steps = []
        total_loss, n_tokens = 0.0, 0
        for t in range(1, T):
            prev_word = tgt[:, t - 1]
            emb = p["Et"][prev_word]                # (B,E)
            proj_h = s_prev @ p["aWh"].T            # (B,H)
            u = np.tanh(proj_s + proj_h[:, None, :])    # (B,S,H)
            e = (u @ p["av"])[:, :, 0]              # (B,S)
            e = np.where(src_mask, e, NEG)
            alpha = _softmax(e, axis=1)             # (B,S)
            context = (alpha[:, :, None] * enc_states).sum(axis=1)  # (B,H)

            x = np.concatenate([emb, context], axis=1)   # (B,E+H)
            concat = np.concatenate([s_prev, x], axis=1)  # (B,H+E+H)
            z = _sigmoid(concat @ p["dWz"].T + p["dbz"])
            r = _sigmoid(concat @ p["dWr"].T + p["dbr"])
            concat_r = np.concatenate([r * s_prev, x], axis=1)
            hh = np.tanh(concat_r @ p["dWh"].T + p["dbh"])
            s_next = (1 - z) * s_prev + z * hh

            cat_out = np.concatenate([s_next, context], axis=1)  # (B,2H)
            logits = cat_out @ p["Wy"].T + p["by"]               # (B,Vt)
            probs = _softmax(logits, axis=1)

            target = tgt[:, t]
            mask_t = (target != mt_data.PAD_ID)
            # cross-entropy on non-pad targets
            idx = np.arange(B)
            total_loss += -np.log(probs[idx, target] + 1e-12)[mask_t].sum()
            n_tokens += mask_t.sum()

            steps.append(dict(prev_word=prev_word, emb=emb, s_prev=s_prev, proj_h=proj_h,
                              u=u, alpha=alpha, context=context, x=x, concat=concat,
                              z=z, r=r, hh=hh, concat_r=concat_r, s_next=s_next,
                              cat_out=cat_out, probs=probs, target=target, mask_t=mask_t))
            s_prev = s_next

        n_tokens = max(int(n_tokens), 1)
        loss = total_loss / n_tokens
        cache = dict(src=src, tgt=tgt, src_mask=src_mask, src_emb=src_emb,
                     enc_states=enc_states, proj_s=proj_s, steps=steps,
                     n_tokens=n_tokens, B=B, S=S, T=T)
        return loss, cache

    # ------------------------- backward -------------------------
    def backward(self, cache):
        p = self.params
        E, H = self.E, self.H
        B, S, T = cache["B"], cache["S"], cache["T"]
        src, src_mask = cache["src"], cache["src_mask"]
        enc_states, proj_s = cache["enc_states"], cache["proj_s"]
        N = cache["n_tokens"]

        g = {k: np.zeros_like(v) for k, v in p.items()}
        d_enc_states = np.zeros((B, S, H))
        ds_next = np.zeros((B, H))

        # ---- decoder backward (reverse over steps) ----
        for st in reversed(cache["steps"]):
            idx = np.arange(B)
            # softmax + cross-entropy gradient (masked, normalised by token count)
            dlogits = st["probs"].copy()
            dlogits[idx, st["target"]] -= 1.0
            dlogits *= st["mask_t"][:, None]
            dlogits /= N

            g["Wy"] += dlogits.T @ st["cat_out"]
            g["by"] += dlogits.sum(axis=0, keepdims=True)
            dcat = dlogits @ p["Wy"]                 # (B,2H)
            ds = dcat[:, :H] + ds_next
            dcontext = dcat[:, H:].copy()

            # decoder GRU backward: s_next = (1-z)*s_prev + z*hh
            z, r, hh = st["z"], st["r"], st["hh"]
            s_prev, x, concat, concat_r = st["s_prev"], st["x"], st["concat"], st["concat_r"]
            dz = ds * (hh - s_prev)
            dhh = ds * z
            ds_prev = ds * (1 - z)
            dhraw = dhh * (1 - hh ** 2)
            g["dWh"] += dhraw.T @ concat_r
            g["dbh"] += dhraw.sum(axis=0, keepdims=True)
            dconcat_r = dhraw @ p["dWh"]
            d_rs = dconcat_r[:, :H]
            dx = dconcat_r[:, H:].copy()
            dr = d_rs * s_prev
            ds_prev += d_rs * r
            dzraw = dz * z * (1 - z)
            drraw = dr * r * (1 - r)
            g["dWz"] += dzraw.T @ concat
            g["dbz"] += dzraw.sum(axis=0, keepdims=True)
            g["dWr"] += drraw.T @ concat
            g["dbr"] += drraw.sum(axis=0, keepdims=True)
            ds_prev += (dzraw @ p["dWz"])[:, :H] + (drraw @ p["dWr"])[:, :H]
            dx += (dzraw @ p["dWz"])[:, H:] + (drraw @ p["dWr"])[:, H:]

            demb = dx[:, :E]
            dcontext += dx[:, E:]
            np.add.at(g["Et"], st["prev_word"], demb)

            # attention backward: context = sum_s alpha_s * enc_states_s
            alpha, u = st["alpha"], st["u"]
            dalpha = (enc_states * dcontext[:, None, :]).sum(axis=2)     # (B,S)
            d_enc_states += alpha[:, :, None] * dcontext[:, None, :]
            # softmax over source positions
            de = alpha * (dalpha - (dalpha * alpha).sum(axis=1, keepdims=True))
            de = np.where(src_mask, de, 0.0)
            # e = u @ av
            g["av"] += np.einsum("bsh,bs->h", u, de)[:, None]
            du = de[:, :, None] * p["av"][:, 0][None, None, :]
            dpre = du * (1 - u ** 2)                                     # (B,S,H)
            # proj_s = enc_states @ aWs.T ; proj_h = s_prev @ aWh.T (broadcast over S)
            g["aWs"] += np.einsum("bsh,bsd->hd", dpre, enc_states)
            d_enc_states += np.einsum("bsh,hd->bsd", dpre, p["aWs"])
            dproj_h = dpre.sum(axis=1)                                   # (B,H)
            g["aWh"] += dproj_h.T @ s_prev
            ds_prev += dproj_h @ p["aWh"]

            ds_next = ds_prev   # flows into s_{t-1} (the previous step's s_next)

        # ---- encoder backward (BPTT over source) ----
        # rebuild per-step h_prev from enc_states (h_prev[0]=0)
        src_emb = cache["src_emb"]
        da_next = np.zeros((B, H))
        for s in reversed(range(S)):
            h_prev = enc_states[:, s - 1, :] if s > 0 else np.zeros((B, H))
            x = src_emb[:, s, :]
            concat = np.concatenate([h_prev, x], axis=1)
            z = _sigmoid(concat @ p["eWz"].T + p["ebz"])
            r = _sigmoid(concat @ p["eWr"].T + p["ebr"])
            concat_r = np.concatenate([r * h_prev, x], axis=1)
            hh = np.tanh(concat_r @ p["eWh"].T + p["ebh"])

            da = d_enc_states[:, s, :] + da_next
            dz = da * (hh - h_prev)
            dhh = da * z
            da_prev = da * (1 - z)
            dhraw = dhh * (1 - hh ** 2)
            g["eWh"] += dhraw.T @ concat_r
            g["ebh"] += dhraw.sum(axis=0, keepdims=True)
            dconcat_r = dhraw @ p["eWh"]
            d_rh = dconcat_r[:, :H]
            dxin = dconcat_r[:, H:].copy()
            dr = d_rh * h_prev
            da_prev += d_rh * r
            dzraw = dz * z * (1 - z)
            drraw = dr * r * (1 - r)
            g["eWz"] += dzraw.T @ concat
            g["ebz"] += dzraw.sum(axis=0, keepdims=True)
            g["eWr"] += drraw.T @ concat
            g["ebr"] += drraw.sum(axis=0, keepdims=True)
            da_prev += (dzraw @ p["eWz"])[:, :H] + (drraw @ p["eWr"])[:, :H]
            dxin += (dzraw @ p["eWz"])[:, H:] + (drraw @ p["eWr"])[:, H:]
            np.add.at(g["Es"], src[:, s], dxin)
            da_next = da_prev

        return g

    # ------------------------- training -------------------------
    def _clip(self, g, max_norm=5.0):
        total = np.sqrt(sum((v ** 2).sum() for v in g.values()))
        if total > max_norm:
            for k in g:
                g[k] *= max_norm / (total + 1e-6)

    def _adam_step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k in self.params:
            self._m[k] = b1 * self._m[k] + (1 - b1) * g[k]
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g[k] ** 2)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, src, tgt, src_dev, tgt_dev, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
        self._init_adam()
        n = len(src)
        best, best_params = float("inf"), None
        for epoch in range(epochs):
            order = np.random.permutation(n)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                loss, cache = self.forward_loss(src[idx], tgt[idx])
                g = self.backward(cache)
                self._clip(g)
                self._adam_step(g, lr)
            dev = self.forward_loss(src_dev, tgt_dev)[0]
            if dev < best:
                best = dev
                best_params = {k: v.copy() for k, v in self.params.items()}
            print(f"epoch {epoch + 1}/{epochs}  dev_loss={dev:.3f}")
        if best_params is not None:
            self.params = best_params
        print(f"best dev_loss={best:.3f}")

    # ------------------------- inference -------------------------
    def translate_ids(self, src_row, max_len=20):
        p = self.params
        E, H = self.E, self.H
        src = src_row[None]                  # (1,S)
        B, S = src.shape
        src_emb = p["Es"][src]
        enc_states = np.zeros((B, S, H))
        h = np.zeros((B, H))
        for s in range(S):
            x = src_emb[:, s, :]
            concat = np.concatenate([h, x], axis=1)
            z = _sigmoid(concat @ p["eWz"].T + p["ebz"])
            r = _sigmoid(concat @ p["eWr"].T + p["ebr"])
            concat_r = np.concatenate([r * h, x], axis=1)
            hh = np.tanh(concat_r @ p["eWh"].T + p["ebh"])
            h = (1 - z) * h + z * hh
            enc_states[:, s, :] = h
        src_mask = src != mt_data.PAD_ID
        proj_s = enc_states @ p["aWs"].T

        s_prev = np.zeros((B, H))
        prev_word = np.array([mt_data.SOS_ID])
        out = []
        for _ in range(max_len):
            emb = p["Et"][prev_word]
            u = np.tanh(proj_s + (s_prev @ p["aWh"].T)[:, None, :])
            e = np.where(src_mask, (u @ p["av"])[:, :, 0], NEG)
            alpha = _softmax(e, axis=1)
            context = (alpha[:, :, None] * enc_states).sum(axis=1)
            x = np.concatenate([emb, context], axis=1)
            concat = np.concatenate([s_prev, x], axis=1)
            z = _sigmoid(concat @ p["dWz"].T + p["dbz"])
            r = _sigmoid(concat @ p["dWr"].T + p["dbr"])
            concat_r = np.concatenate([r * s_prev, x], axis=1)
            hh = np.tanh(concat_r @ p["dWh"].T + p["dbh"])
            s_prev = (1 - z) * s_prev + z * hh
            logits = np.concatenate([s_prev, context], axis=1) @ p["Wy"].T + p["by"]
            nxt = int(logits.argmax(axis=1)[0])
            if nxt == mt_data.EOS_ID:
                break
            out.append(nxt)
            prev_word = np.array([nxt])
        return out

    def save(self, path):
        np.savez(path, E=self.E, H=self.H, **self.params)

    @classmethod
    def load(cls, path):
        m = cls()
        npz = np.load(path)
        m.E, m.H = int(npz["E"]), int(npz["H"])
        m.params = {k: npz[k] for k in npz.files if k not in ("E", "H")}
        return m


# --------------------------- gradient check ---------------------------

def gradient_check():
    rng = np.random.default_rng(0)
    Vs, Vt, E, H, B, S, T = 9, 8, 6, 5, 3, 4, 5
    model = ManualSeq2SeqAttention(Vs, Vt, emb=E, hid=H, seed=1)
    src = rng.integers(0, Vs, size=(B, S)); src[0, -1] = mt_data.PAD_ID   # test mask
    tgt = rng.integers(1, Vt, size=(B, T)); tgt[1, -1] = mt_data.PAD_ID   # test target mask

    _, cache = model.forward_loss(src, tgt)
    grads = model.backward(cache)

    # Directional-derivative check: compare the analytic gradient against a finite
    # difference along a single random direction over ALL parameters at once. This
    # is far more robust than per-element checks for parameters buried deep in the
    # recurrence (where per-element finite differences accumulate numerical noise).
    dirs = {k: rng.standard_normal(v.shape) for k, v in model.params.items()}
    analytic = sum(float((grads[k] * dirs[k]).sum()) for k in model.params)

    eps = 1e-5
    saved = {k: model.params[k].copy() for k in model.params}
    for k in model.params:
        model.params[k] = saved[k] + eps * dirs[k]
    lp = model.forward_loss(src, tgt)[0]
    for k in model.params:
        model.params[k] = saved[k] - eps * dirs[k]
    lm = model.forward_loss(src, tgt)[0]
    for k in model.params:
        model.params[k] = saved[k]
    numeric = (lp - lm) / (2 * eps)

    rel = abs(numeric - analytic) / max(1e-9, abs(numeric) + abs(analytic))
    print(f"gradient check (directional): analytic={analytic:.6f} numeric={numeric:.6f} "
          f"rel_err={rel:.2e}  ({'PASS' if rel < 1e-5 else 'FAIL'})")
    return rel


def train_and_save(epochs=EPOCHS):
    cfg = mt_data.load_config()
    src, tgt = mt_data.load_split("train")
    dv_s, dv_t = mt_data.load_split("dev")
    m = ManualSeq2SeqAttention(cfg["vocab_src"], cfg["vocab_tgt"])
    np.random.seed(SEED)
    m.fit(src.astype(np.int64), tgt.astype(np.int64),
          dv_s.astype(np.int64), dv_t.astype(np.int64), epochs=epochs)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    m.save(ARTIFACTS / "manual_seq2seq.npz")
    print(f"saved -> {ARTIFACTS / 'manual_seq2seq.npz'}")
    return m


if __name__ == "__main__":
    import sys
    if "train" in sys.argv:
        train_and_save()
    else:
        gradient_check()
