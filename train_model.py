import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from preprocess import clean_text

# Load the raw text and clean it with the SAME function the API uses
df = pd.read_csv("spam_cleaned.xls")[["text", "label_num"]].dropna()
df = df.drop_duplicates(subset="text")
df["clean_text"] = df["text"].astype(str).apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_num"],
    test_size=0.2, stratify=df["label_num"], random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# Evaluate once on the held-out test set
y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "train_size": len(X_train), "test_size": len(X_test),
}

joblib.dump(model, "spam_model.joblib")
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)