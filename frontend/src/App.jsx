import { useEffect, useState } from "react";

// project-specific labels (the Transformer frontend only changes these three)
const TITLE = "GRU + Attention";
const SUBTITLE = "French → English · seq2seq with Bahdanau attention";
const TRAIN_HINT = "python seq2seq_pytorch.py   (and: python seq2seq_manual.py train)";

const API_BASE = import.meta.env.VITE_API_URL || "/api";
const ICON = { pytorch: "🔥", tensorflow: "🧠", manual: "🧮" };
const SAMPLES = [
  "je vous remercie .",
  "merci beaucoup .",
  "le parlement européen",
  "nous devons agir maintenant .",
];

export default function App() {
  const [text, setText] = useState(SAMPLES[0]);
  const [results, setResults] = useState([]);
  const [submitted, setSubmitted] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(null);
  const [bleu, setBleu] = useState(null);   // { model: score } once computed

  useEffect(() => {
    fetch(`${API_BASE}/info`)
      .then((r) => r.json())
      .then((d) => setLoaded(d.models || []))
      .catch(() => setLoaded([]));
    // BLEU is scored on the test set server-side (can take a few seconds);
    // fetch in the background and show it when ready.
    fetch(`${API_BASE}/bleu`)
      .then((r) => r.json())
      .then((d) => setBleu(d.bleu || {}))
      .catch(() => setBleu({}));
  }, []);

  async function runTranslate() {
    const t = text.trim();
    if (!t) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t }),
      });
      if (!res.ok) throw new Error(`server returned ${res.status}`);
      const data = await res.json();
      setResults(data.results || []);
      setSubmitted(data.text || t);
    } catch (e) {
      setError(`Could not reach the backend (${e.message}). Is it running on :8000?`);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runTranslate();
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>🌍 <span className="accent">{TITLE}</span></h1>
        <p className="sub">{SUBTITLE}</p>
      </header>

      {loaded && loaded.length === 0 && (
        <div className="notice">
          No trained models found. Train at least one, then restart the backend:
          <pre>{TRAIN_HINT}</pre>
        </div>
      )}

      <section className="input-card">
        <label className="lang-tag">French</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Tapez une phrase en français…"
          rows={3}
        />
        <div className="controls">
          <button className="go" onClick={runTranslate} disabled={loading}>
            {loading ? <><span className="spinner" /> Translating…</> : "Translate →"}
          </button>
          <span className="hint">⌘/Ctrl + Enter</span>
          <div className="samples">
            {SAMPLES.map((s, i) => (
              <button key={i} className="sample" onClick={() => setText(s)}>
                ex {i + 1}
              </button>
            ))}
          </div>
        </div>
        {loaded && loaded.length > 0 && (
          <div className="loaded">models: {loaded.join(" · ")}</div>
        )}
      </section>

      {error && <div className="error">{error}</div>}

      {submitted && !error && results.length > 0 && (
        <section className="results fade-in">
          <p className="submitted">“{submitted}”</p>
          {results.map((r) => (
            <div key={r.model} className="card">
              <span className="model">
                {ICON[r.model] || "•"} {r.model}
                {bleu && bleu[r.model] != null && (
                  <span className="bleu" title="corpus BLEU on the test set">
                    BLEU {bleu[r.model]}
                  </span>
                )}
                {bleu === null && <span className="bleu pending">BLEU…</span>}
              </span>
              <span className="translation">{r.translation || "—"}</span>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
