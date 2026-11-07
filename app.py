import os
import json
import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from preprocess import clean_text

BASE = Path(__file__).parent

# Loaded once per server instance, then reused for every request (fast + scalable)
model = joblib.load(BASE / "spam_model.joblib")
metrics = json.loads((BASE / "metrics.json").read_text())

app = FastAPI(title="SMS Spam Classifier API")


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def get_metrics():
    """Test-set results saved during training."""
    return metrics


@app.post("/api/predict")
def predict(msg: Message):
    cleaned = clean_text(msg.text)
    spam_prob = float(model.predict_proba([cleaned])[0][1])

    # Explain the decision: which words pushed the score the most
    tfidf = model.named_steps["tfidf"]
    coefs = model.named_steps["clf"].coef_[0]
    names = tfidf.get_feature_names_out()
    vec = tfidf.transform([cleaned])
    contributions = [(names[i], float(vec[0, i] * coefs[i])) for i in vec.nonzero()[1]]
    top = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "prediction": "spam" if spam_prob >= 0.5 else "ham",
        "spam_probability": round(spam_prob, 4),
        "top_words": [{"word": w, "effect": "spam" if s > 0 else "ham"} for w, s in top],
    }


# For local testing only. On Vercel the public/ folder is served by the CDN automatically.
if not os.environ.get("VERCEL"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")