"""Learned intent classifier: TF-IDF character n-grams + multinomial logistic regression."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from iris_bot.nlu.keywords import normalize


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=1, sublinear_tf=True,
                                  preprocessor=normalize)),
        ("clf", LogisticRegression(C=30.0, max_iter=2000, class_weight="balanced")),
    ])


class IntentModel:
    name = "tfidf_char_logreg"

    def __init__(self, pipeline: Pipeline, meta: dict[str, Any] | None = None):
        self.pipeline = pipeline
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path) -> IntentModel:
        obj = joblib.load(path)
        return cls(obj["pipeline"], obj.get("meta"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "meta": self.meta}, path, compress=3)

    def predict(self, text: str) -> tuple[str, float]:
        proba = self.pipeline.predict_proba([text])[0]
        i = int(proba.argmax())
        return str(self.pipeline.classes_[i]), float(proba[i])

    def predict_many(self, texts: list[str]) -> list[str]:
        return [str(x) for x in self.pipeline.predict(texts)]
