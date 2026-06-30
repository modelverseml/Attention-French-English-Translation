"""
French -> English translator: GRU encoder-decoder with Bahdanau attention, in
TensorFlow/Keras. Same architecture as seq2seq_pytorch.py, trained with a custom
GradientTape loop (teacher forcing + masked cross-entropy) and greedy decoding.

Run:  python seq2seq_tensorflow.py
"""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

import mt_data

ARTIFACTS = Path(__file__).resolve().parent.parent / "data" / "model_artifacts"
EMB_DIM = 256
HID = 256
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab, emb_dim, hid):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab, emb_dim, mask_zero=False)
        self.gru = tf.keras.layers.GRU(hid, return_sequences=True, return_state=True)

    def call(self, src):
        return self.gru(self.embed(src))   # outputs (B,S,hid), state (B,hid)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, hid):
        super().__init__()
        self.Wh = tf.keras.layers.Dense(hid, use_bias=False)
        self.Ws = tf.keras.layers.Dense(hid, use_bias=False)
        self.v = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, dec_h, enc_out, src_mask):
        # dec_h (B,hid), enc_out (B,S,hid), src_mask (B,S) bool
        scores = self.v(tf.nn.tanh(
            tf.expand_dims(self.Wh(dec_h), 1) + self.Ws(enc_out)))   # (B,S,1)
        scores = tf.squeeze(scores, -1)
        scores = tf.where(src_mask, scores, tf.fill(tf.shape(scores), -1e9))
        attn = tf.nn.softmax(scores, axis=1)                        # (B,S)
        context = tf.reduce_sum(tf.expand_dims(attn, -1) * enc_out, axis=1)
        return context, attn


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab, emb_dim, hid):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab, emb_dim, mask_zero=False)
        self.attn = BahdanauAttention(hid)
        self.gru = tf.keras.layers.GRU(hid, return_sequences=True, return_state=True)
        self.out = tf.keras.layers.Dense(vocab)

    def step(self, prev_word, state, enc_out, src_mask):
        emb = self.embed(prev_word)                                 # (B,E)
        context, attn = self.attn(state, enc_out, src_mask)
        gru_in = tf.expand_dims(tf.concat([emb, context], axis=1), 1)
        out, state = self.gru(gru_in, initial_state=state)
        logits = self.out(tf.concat([tf.squeeze(out, 1), context], axis=1))
        return logits, state, attn


class Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_src, vocab_tgt, emb_dim=EMB_DIM, hid=HID):
        super().__init__()
        self.encoder = Encoder(vocab_src, emb_dim, hid)
        self.decoder = Decoder(vocab_tgt, emb_dim, hid)

    def call(self, src, tgt, teacher_forcing=1.0, training=False):
        enc_out, state = self.encoder(src)
        src_mask = tf.not_equal(src, mt_data.PAD_ID)
        T = tgt.shape[1]
        logits_seq = []
        prev = tgt[:, 0]
        for t in range(1, T):
            logits, state, _ = self.decoder.step(prev, state, enc_out, src_mask)
            logits_seq.append(logits)
            pred = tf.cast(tf.argmax(logits, -1), tgt.dtype)
            if training:
                # graph-safe teacher forcing: a Python `if` on a tf.Tensor isn't
                # allowed under @tf.function, so choose with tf.where instead.
                use_tf = tf.random.uniform([]) < teacher_forcing
                prev = tf.where(use_tf, tgt[:, t], pred)
            else:
                prev = pred
        return tf.stack(logits_seq, axis=1)                         # (B,T-1,vocab)


def masked_loss(tgt_out, logits):
    # sparse CE, ignoring pad positions
    ce = tf.keras.losses.sparse_categorical_crossentropy(tgt_out, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(tgt_out, mt_data.PAD_ID), ce.dtype)
    return tf.reduce_sum(ce * mask) / tf.reduce_sum(mask)


def train():
    tf.random.set_seed(SEED)
    cfg = mt_data.load_config()
    src_tr, tgt_tr = mt_data.load_split("train")
    src_dv, tgt_dv = mt_data.load_split("dev")
    train_ds = (tf.data.Dataset.from_tensor_slices((src_tr, tgt_tr))
                .shuffle(4096).batch(BATCH_SIZE))

    model = Seq2Seq(cfg["vocab_src"], cfg["vocab_tgt"])
    opt = tf.keras.optimizers.Adam(LR)

    @tf.function
    def train_step(src, tgt):
        with tf.GradientTape() as tape:
            logits = model(src, tgt, teacher_forcing=0.5, training=True)
            loss = masked_loss(tgt[:, 1:], logits)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_dev = float("inf")
    for epoch in range(EPOCHS):
        for src, tgt in train_ds:
            train_step(src, tgt)
        dev_logits = model(src_dv, tgt_dv, teacher_forcing=1.0, training=False)
        dev = float(masked_loss(tgt_dv[:, 1:], dev_logits))
        if dev < best_dev:
            best_dev = dev
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(ARTIFACTS / "tensorflow_seq2seq.weights.h5"))
        print(f"epoch {epoch + 1}/{EPOCHS}  dev_loss={dev:.3f}")
    print(f"best dev_loss={best_dev:.3f}  saved -> tensorflow_seq2seq.weights.h5")
    return model


def translate(model, sentence, fr_vocab, en_inv, cfg, max_len=20):
    src = mt_data.encode_source(sentence, fr_vocab, cfg["src_len"])[None].astype("int32")
    enc_out, state = model.encoder(src)
    src_mask = tf.not_equal(src, mt_data.PAD_ID)
    prev = tf.constant([mt_data.SOS_ID], dtype=tf.int32)
    ids = []
    for _ in range(max_len):
        logits, state, _ = model.decoder.step(prev, state, enc_out, src_mask)
        prev = tf.cast(tf.argmax(logits, -1), tf.int32)
        nxt = int(prev.numpy()[0])
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
