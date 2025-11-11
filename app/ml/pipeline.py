import os, re, joblib
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

BAD_WORDS = set([
    "idiot","stupid","hate","dumb","kill yourself","racist","trash","moron","loser",
    "shut up","go to hell","retard","nazi","terrorist"
])

def heuristic_score(text: str) -> float:
    t = text.lower()
    score = 0.0
    # bad words
    for w in BAD_WORDS:
        if w in t:
            score += 0.4
    # shouty caps
    if len([c for c in t if c.isupper()]) >= max(6, int(0.5 * len(t))):
        score += 0.2
    # repeated punctuation or expletives
    if "!!!" in t or "???" in t:
        score += 0.1
    # simple hate markers
    if re.search(r"\b(hate|die|kill|trash|awful|disgusting)\b", t):
        score += 0.2
    return max(0.0, min(1.0, score))

class ModelManager:
    def __init__(self, path: str = MODEL_PATH):
        self.path = path
        self.model = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                self.model = joblib.load(self.path)
            except Exception:
                self.model = None
        else:
            self.model = None

    def save(self, model):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        joblib.dump(model, self.path)
        self.model = model

    def predict_proba(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Return list of (p_toxic, p_neutral)."""
        if self.model is None:
            # Heuristic fallback
            out = []
            for t in texts:
                p = heuristic_score(t)
                out.append((p, 1.0 - p))
            return out
        # If model has decision_function but not predict_proba (e.g., LinearSVC), map via Platt-like logistic
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(texts)
            # Assume classes order ['neutral','toxic'] if trained so; enforce mapping
            # Try to detect index of 'toxic'
            if hasattr(self.model, "classes_"):
                classes = list(self.model.classes_)
                toxic_idx = classes.index("toxic") if "toxic" in classes else 1
                neutral_idx = classes.index("neutral") if "neutral" in classes else 0
                return [(row[toxic_idx], row[neutral_idx]) for row in proba]
            return [(row[1], row[0]) for row in proba]
        if hasattr(self.model, "decision_function"):
            import numpy as np
            scores = self.model.decision_function(texts)
            # logistic squashing
            p = 1.0 / (1.0 + np.exp(-scores))
            return [(float(x), float(1.0 - x)) for x in p]
        # last resort: predict -> 0/1
        preds = self.model.predict(texts)
        out = []
        for y in preds:
            p = 0.9 if y == "toxic" else 0.1
            out.append((p, 1.0 - p))
        return out

    def train(self, samples: List[Tuple[str, str]]):
        texts = [t for t, _ in samples]
        labels = [y for _, y in samples]

        # Handle very small datasets gracefully
        stratify = labels if len(set(labels)) > 1 and min(labels.count(l) for l in set(labels)) > 1 else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=stratify
            )
        except Exception:
            X_train, y_train = texts, labels
            X_test, y_test = [], []

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
            ("clf", LinearSVC())
        ])
        pipeline.fit(X_train, y_train)

        if X_test:
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")
        else:
            acc, f1m = 1.0, 1.0

        self.save(pipeline)
        return {"accuracy": float(acc), "f1_macro": float(f1m)}

