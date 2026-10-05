"""Train and evaluate the IRIS intent classifier.

Compares, on the held-out test split (unseen template families):
  1. keyword rules baseline
  2. TF-IDF char n-grams + logistic regression (the shipped model)
  3. LLM zero-shot (DeepSeek Flash by default) - only with --llm and DEEPSEEK_API_KEY set
Also reports grouped cross-validation on train (GroupKFold by family) and language-detection accuracy.
Writes models/intent.joblib and models/intent_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iris_bot.nlu import INTENTS, KeywordClassifier, detect_language  # noqa: E402
from iris_bot.nlu.model import IntentModel, build_pipeline  # noqa: E402

DATA = ROOT / "eval" / "intents"
MODEL_PATH = ROOT / "models" / "intent.joblib"
METRICS_PATH = ROOT / "models" / "intent_metrics.json"


def load(split: str) -> list[dict]:
    with open(DATA / f"{split}.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def scores(y_true: list[str], y_pred: list[str], langs: list[str]) -> dict:
    out = {"macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=list(INTENTS), zero_division=0), 4),
           "accuracy": round(accuracy_score(y_true, y_pred), 4), "n": len(y_true)}
    for lang in sorted(set(langs)):
        idx = [i for i, g in enumerate(langs) if g == lang]
        yt, yp = [y_true[i] for i in idx], [y_pred[i] for i in idx]
        out[lang] = {"macro_f1": round(f1_score(yt, yp, average="macro", labels=sorted(set(yt)), zero_division=0), 4),
                     "accuracy": round(accuracy_score(yt, yp), 4), "n": len(idx)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate LLM zero-shot (needs the provider API key, DeepSeek by default)")
    args = ap.parse_args()

    train, test = load("train"), load("test")
    Xtr, ytr = [r["text"] for r in train], [r["intent"] for r in train]
    Xte, yte, lte = [r["text"] for r in test], [r["intent"] for r in test], [r["lang"] for r in test]

    # Grouped CV on train: folds never share a template family.
    groups = [r["family"] for r in train]
    cv = []
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(Xtr, ytr, groups):
        p = build_pipeline().fit([Xtr[i] for i in tr_idx], [ytr[i] for i in tr_idx])
        cv.append(f1_score([ytr[i] for i in va_idx], p.predict([Xtr[i] for i in va_idx]), average="macro"))

    pipe = build_pipeline().fit(Xtr, ytr)
    model = IntentModel(pipe, {"train_rows": len(train), "intents": list(INTENTS)})
    rules = KeywordClassifier()

    results = {
        "keyword_rules": scores(yte, rules.predict_many(Xte), lte),
        "tfidf_char_logreg": scores(yte, model.predict_many(Xte), lte),
    }
    # Confidence-gated behaviour as used by the agent (below threshold -> clarification).
    conf = [model.predict(x)[1] for x in Xte]
    results["tfidf_char_logreg"]["share_above_0.6_confidence"] = round(float(np.mean([c >= 0.6 for c in conf])), 4)
    above = [i for i, c in enumerate(conf) if c >= 0.6]
    results["tfidf_char_logreg"]["accuracy_when_above_0.6"] = round(
        accuracy_score([yte[i] for i in above], [model.predict(Xte[i])[0] for i in above]), 4) if above else None

    if args.llm:
        from iris_bot.config import get_settings
        from iris_bot.llm import build_llm
        s = get_settings()
        if not s.provider_api_key:
            results["llm_zero_shot"] = f"not run: API key for provider '{s.provider}' not set"
        else:
            llm = build_llm(s)
            preds, cost = [], 0.0
            for x, lang in zip(Xte, lte):
                res, u = llm.classify_intent(x, lang)
                preds.append(res[0] if res else "out_of_scope")
                cost += u.cost_usd
            results["llm_zero_shot"] = scores(yte, preds, lte) | {"model": s.classifier_model,
                                                                  "cost_usd": round(cost, 4)}
    else:
        results["llm_zero_shot"] = "not run (use --llm with DEEPSEEK_API_KEY set)"

    lang_pred = [detect_language(x)[0] for x in Xte]
    metrics = {
        "data": {"train": len(train), "test": len(test), "train_by_lang": dict(Counter(r["lang"] for r in train)),
                 "test_by_lang": dict(Counter(lte)), "test_families": len({r["family"] for r in test}),
                 "split": "held-out template families (one per intent x language)"},
        "grouped_cv_macro_f1_train": {"mean": round(float(np.mean(cv)), 4), "std": round(float(np.std(cv)), 4)},
        "test": results,
        "language_detection_accuracy": round(accuracy_score(lte, lang_pred), 4),
    }
    model.meta["metrics"] = metrics["test"]["tfidf_char_logreg"]
    model.save(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    errors = [(x, y, p) for x, y, p in zip(Xte, yte, model.predict_many(Xte)) if y != p]
    print(f"\nmodel errors on test ({len(errors)}):")
    for x, y, p in errors:
        print(f"  [{y} -> {p}] {x}")
    print(f"\nsaved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
