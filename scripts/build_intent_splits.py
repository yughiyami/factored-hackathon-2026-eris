"""Build eval/intents/train.jsonl and test.jsonl from the team-written utterances.yaml.

Split rule (prevents template leakage): every utterance of a family whose name starts with "test_" goes
to test; all other families go to train. A family is one phrasing template, so no template is shared
between splits. One family per intent x language is held out.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval" / "intents" / "utterances.yaml"
OUT = ROOT / "eval" / "intents"


def main() -> None:
    data = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    rows = {"train": [], "test": []}
    for intent, by_lang in data.items():
        for lang, families in by_lang.items():
            for family, items in families.items():
                split = "test" if family.startswith("test_") else "train"
                for item in items:
                    variant, text = item.split(":", 1)
                    rows[split].append({"text": text.strip(), "intent": intent, "lang": lang,
                                        "variant": variant.strip(), "family": f"{intent}/{lang}/{family}"})
    train_fams = {r["family"] for r in rows["train"]}
    assert not train_fams & {r["family"] for r in rows["test"]}, "family leakage"
    assert not {r["text"].lower() for r in rows["train"]} & {r["text"].lower() for r in rows["test"]}, "dup text"
    for split, items in rows.items():
        path = OUT / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in items:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items)} rows  by lang {dict(Counter(r['lang'] for r in items))}  "
              f"by variant {dict(Counter(r['variant'] for r in items))}")


if __name__ == "__main__":
    main()
