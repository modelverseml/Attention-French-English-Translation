"""
FastAPI backend for the Attention (GRU+attention seq2seq) translator.

Loads whatever trained models exist (via translator.py) and exposes /translate
for the React frontend. Run (from this folder, project venv active):

    uvicorn app:app --reload --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# translator.py + the model modules live in ../ (the code/ folder)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import translator

BLEU_N = 150   # test sentences to score for the /bleu badge (cached after first call)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading trained models …")
    app.state.models = translator.load_models()
    print("ready:", list(app.state.models) or "(none trained yet)")
    yield


app = FastAPI(title="FR→EN Attention API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class TranslateRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {"models": list(app.state.models)}


@app.get("/bleu")
def bleu():
    # corpus BLEU per model over a slice of the test set. computed once and cached
    # (translating the test set is the slow part, especially for the manual model).
    if getattr(app.state, "bleu", None) is None:
        if not app.state.models:
            app.state.bleu = {}
        else:
            from evaluate_bleu import score_models
            scores = score_models(app.state.models, n=BLEU_N)
            app.state.bleu = {k: round(v * 100, 2) for k, v in scores.items()}
    return {"bleu": app.state.bleu, "n": BLEU_N}


@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text or not app.state.models:
        return {"text": text, "results": []}
    outs = translator.translate_all(text, app.state.models)
    return {"text": text,
            "results": [{"model": k, "translation": v} for k, v in outs.items()]}
