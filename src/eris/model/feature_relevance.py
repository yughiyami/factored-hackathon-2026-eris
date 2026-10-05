"""Variable relevance for the IRIS KPIs.

One LightGBM model per KPI, each compared against a simple, explainable baseline on the
same out-of-time test set:

  target           KPI                      feature set      baseline
  csat_top2        Customer satisfaction    full (post)      rate by was_resolved
  fcr              First-contact resolution pre-contact      rate by contact_reason
  escalated        % redirection            pre-contact      rate by contact_reason
  cycle_time       Cycle time (s)           pre-contact      mean by contact_reason

"pre-contact" = only what the system knows when the contact starts (routing decision);
"full" also includes in-contact signals and outcomes, to explain what drives CSAT.

Split is temporal to prevent leakage: train < 2025-07-01 <= valid < 2026-01-01 <= test.
Relevance is reported three ways: LightGBM gain, mean |SHAP| and permutation importance
on the held-out test set (the last one is the most honest about real predictive value).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "data" / "gold" / "interaction_features.parquet"
OUT = ROOT / "reports"
SEED = 42

PRE_CONTACT = [
    "interaction_type", "channel", "contact_reason", "hour_of_day", "day_of_week", "wait_time_seconds",
    "country", "segment", "customer_status", "document_type", "gender", "customer_accent",
    "credit_score", "estimated_monthly_income", "customer_age", "tenure_days", "accepts_marketing",
    "agent_type", "experience_level", "agent_specialty", "work_shift", "agent_accent", "accent_match",
    "agent_tenure_days", "prior_contacts_30d", "prior_contacts_90d", "prior_escalations_90d",
    "prior_unresolved_90d", "days_since_last_contact", "same_reason_as_last", "prior_complaints_90d",
    "n_products", "n_credit_cards", "n_loans", "max_days_past_due", "n_blocked_products",
]
IN_CONTACT = ["duration_seconds", "sentiment_score", "detected_sentiment", "requires_followup",
              "was_resolved", "was_escalated"]

TASKS = {
    "csat_top2": dict(kind="clf", features=PRE_CONTACT + IN_CONTACT, baseline_by="was_resolved"),
    "fcr": dict(kind="clf", features=PRE_CONTACT, baseline_by="contact_reason"),
    "escalated": dict(kind="clf", features=PRE_CONTACT, baseline_by="contact_reason"),
    "cycle_time": dict(kind="reg", features=PRE_CONTACT, baseline_by="contact_reason"),
}


def load() -> pd.DataFrame:
    df = pd.read_parquet(GOLD)
    df["csat_top2"] = np.where(df["csat_score"].isna(), np.nan, (df["csat_score"] >= 3).astype(float))
    df["fcr"] = df["was_resolved"].astype("float")
    df["escalated"] = df["was_escalated"].astype("float")
    df["cycle_time"] = df["duration_seconds"]  # wait time is a feature, not part of this target
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("float")
        elif pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = df[c].astype("category")
        elif str(df[c].dtype) == "boolean":
            df[c] = df[c].astype("float")
    df["split"] = np.select(
        [df["interaction_date"] < "2025-07-01", df["interaction_date"] < "2026-01-01"],
        ["train", "valid"], "test")
    return df


def baseline(train: pd.DataFrame, test: pd.DataFrame, target: str, by: str) -> np.ndarray:
    table = train.groupby(by, observed=True)[target].mean()
    return test[by].map(table).astype(float).fillna(train[target].mean()).to_numpy()


def score(kind: str, y: np.ndarray, p: np.ndarray) -> dict:
    if kind == "clf":
        return {"roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
                "positive_rate": round(float(y.mean()), 4)}
    return {"mae": round(mean_absolute_error(y, p), 2), "target_mean": round(float(y.mean()), 2)}


def run_task(df: pd.DataFrame, target: str, cfg: dict) -> tuple[dict, pd.DataFrame]:
    d = df[df[target].notna()]
    tr, va, te = (d[d.split == s] for s in ("train", "valid", "test"))
    X = cfg["features"]
    params = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    model = lgb.LGBMClassifier(**params) if cfg["kind"] == "clf" else lgb.LGBMRegressor(objective="l1", **params)
    model.fit(tr[X], tr[target], eval_set=[(va[X], va[target])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = model.predict_proba(te[X])[:, 1] if cfg["kind"] == "clf" else model.predict(te[X])
    base = baseline(tr, te, target, cfg["baseline_by"])
    result = {
        "n_train": len(tr), "n_valid": len(va), "n_test": len(te), "best_iteration": model.best_iteration_,
        "baseline": {"by": cfg["baseline_by"], **score(cfg["kind"], te[target].to_numpy(), base)},
        "lightgbm": score(cfg["kind"], te[target].to_numpy(), pred),
    }

    sample = te.sample(min(20000, len(te)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(sample[X])
    sv = sv[1] if isinstance(sv, list) else sv
    perm = permutation_importance(model, sample[X], sample[target], n_repeats=3, random_state=SEED,
                                  scoring="roc_auc" if cfg["kind"] == "clf" else "neg_mean_absolute_error")
    imp = pd.DataFrame({
        "target": target, "feature": X,
        "gain": model.booster_.feature_importance("gain"),
        "shap_mean_abs": np.abs(sv).mean(axis=0),
        "permutation": perm.importances_mean,
        "permutation_std": perm.importances_std,
    })
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp["shap_share"] = imp["shap_mean_abs"] / imp["shap_mean_abs"].sum()
    return result, imp.sort_values("permutation", ascending=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()
    results, imps = {}, []
    for target, cfg in TASKS.items():
        res, imp = run_task(df, target, cfg)
        results[target] = res
        imps.append(imp)
        print(f"\n### {target}: {json.dumps(res)}")
        print(imp.head(10)[["feature", "permutation", "shap_share", "gain_share"]].to_string(index=False))
    (OUT / "model_results.json").write_text(json.dumps(results, indent=2))
    pd.concat(imps).to_csv(OUT / "feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
