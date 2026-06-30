"""
French -> English translator: GRU encoder-decoder with Bahdanau (additive)
attention, in PyTorch.

    encoder:   embed(src) -> GRU -> per-token states  H = (B, S, h)
    attention: for each decode step, score the decoder state against every H_s,
               softmax over source positions (pad masked) -> context vector
    decoder:   GRU over [embed(prev_word), context] -> linear -> vocab logits

Trained with teacher forcing + cross-entropy (padding ignored); translates with
greedy decoding. Run:  python seq2seq_pytorch.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import mt_data

ARTIFACTS = Path(__file__).resolve().parent.parent / "data" / "model_artifacts"
EMB_DIM = 256
HID = 256
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim, hid):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=mt_data.PAD_ID)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True)

    def forward(self, src):
        # src: (B, S) -> outputs (B, S, hid), hidden (1, B, hid)
        return self.gru(self.embed(src))


class BahdanauAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.Wh = nn.Linear(hid, hid, bias=False)   # decoder state
        self.Ws = nn.Linear(hid, hid, bias=False)   # encoder states
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, dec_h, enc_out, src_mask):
        # dec_h (B, hid), enc_out (B, S, hid), src_mask (B, S) True=real token
        scores = self.v(torch.tanh(self.Wh(dec_h).unsqueeze(1) + self.Ws(enc_out))).squeeze(-1)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn = F.softmax(scores, dim=1)              # (B, S)
        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # (B, hid)
        return context, attn


class Decoder(nn.Module):
    def __init__(self, vocab, emb_dim, hid):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=mt_data.PAD_ID)
        self.attn = BahdanauAttention(hid)
        self.gru = nn.GRU(emb_dim + hid, hid, batch_first=True)
        self.out = nn.Linear(hid + hid, vocab)

    def step(self, prev_word, hidden, enc_out, src_mask):
        # one decode step. prev_word (B,), hidden (1,B,hid)
        emb = self.embed(prev_word).unsqueeze(1)              # (B,1,E)
        context, attn = self.attn(hidden[-1], enc_out, src_mask)
        gru_in = torch.cat([emb, context.unsqueeze(1)], dim=2)
        out, hidden = self.gru(gru_in, hidden)               # out (B,1,hid)
        logits = self.out(torch.cat([out.squeeze(1), context], dim=1))
        return logits, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, emb_dim=EMB_DIM, hid=HID):
        super().__init__()
        self.encoder = Encoder(vocab_src, emb_dim, hid)
        self.decoder = Decoder(vocab_tgt, emb_dim, hid)

    def forward(self, src, tgt, teacher_forcing=1.0):
        # returns logits (B, T-1, vocab) predicting tgt[:,1:]
        enc_out, hidden = self.encoder(src)
        src_mask = src != mt_data.PAD_ID
        B, T = tgt.shape
        logits = []
        prev = tgt[:, 0]                                     # <sos>
        for t in range(1, T):
            step_logits, hidden, _ = self.decoder.step(prev, hidden, enc_out, src_mask)
            logits.append(step_logits)
            use_tf = torch.rand(1).item() < teacher_forcing
            prev = tgt[:, t] if use_tf else step_logits.argmax(-1)
        return torch.stack(logits, dim=1)


def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt, teacher_forcing=1.0)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            total += loss.item() * len(src); n += len(src)
    return total / max(n, 1)


def train():
    torch.manual_seed(SEED)
    cfg = mt_data.load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loader(name, shuffle):
        src, tgt = mt_data.load_split(name)
        ds = TensorDataset(torch.from_numpy(src.astype(np.int64)),
                           torch.from_numpy(tgt.astype(np.int64)))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader, dev_loader = loader("train", True), loader("dev", False)
    model = Seq2Seq(cfg["vocab_src"], cfg["vocab_tgt"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=mt_data.PAD_ID)

    best_dev, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(src, tgt, teacher_forcing=0.5)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        dev = evaluate_loss(model, dev_loader, loss_fn, device)
        if dev < best_dev:
            best_dev = dev
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{EPOCHS}  dev_loss={dev:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best dev_loss={best_dev:.3f}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS / "pytorch_seq2seq.pt")
    print(f"saved -> {ARTIFACTS / 'pytorch_seq2seq.pt'}")
    return model


@torch.no_grad()
def translate(model, sentence, fr_vocab, en_inv, cfg, max_len=20):
    model.eval()
    src = torch.from_numpy(mt_data.encode_source(sentence, fr_vocab, cfg["src_len"]))[None]
    enc_out, hidden = model.encoder(src)
    src_mask = src != mt_data.PAD_ID
    prev = torch.tensor([mt_data.SOS_ID])
    ids = []
    for _ in range(max_len):
        logits, hidden, _ = model.decoder.step(prev, hidden, enc_out, src_mask)
        prev = logits.argmax(-1)
        nxt = int(prev.item())
        if nxt == mt_data.EOS_ID:
            break
        ids.append(nxt)
    return mt_data.decode_target(ids, en_inv)


if __name__ == "__main__":
    model = train()
    fr_vocab, _, en_inv = mt_data.load_vocabs()
    cfg = mt_data.load_config()
    print("\n=== sample translations (greedy) ===")
    for s in ["je vous remercie .", "le parlement européen", "nous devons agir maintenant ."]:
        print(f"  fr: {s}\n  en: {translate(model, s, fr_vocab, en_inv, cfg)}\n")
