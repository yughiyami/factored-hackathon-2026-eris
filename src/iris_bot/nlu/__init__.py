"""NLU: intent classification (keyword baseline, learned model, LLM fallback) + language detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iris_bot.nlu.keywords import INTENTS, KeywordClassifier
from iris_bot.nlu.language import detect_language

__all__ = ["INTENTS", "KeywordClassifier", "detect_language", "IntentResult", "load_classifier"]


@dataclass
class IntentResult:
    intent: str
    confidence: float
    source: str  # keyword_rules | tfidf_char_logreg | llm


def load_classifier(path: Path, fallback_to_rules: bool = True):
    """Load the trained model; fall back to keyword rules if the artifact is missing."""
    if path.exists():
        from iris_bot.nlu.model import IntentModel
        return IntentModel.load(path)
    if fallback_to_rules:
        return KeywordClassifier()
    raise FileNotFoundError(path)
