#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "labeling_sample.csv"
DEFAULT_OUTPUT = ROOT / "data" / "labeled_messages.csv"


def auto_label(df: pd.DataFrame) -> pd.DataFrame:
    required = {"risk_score", "profanity_count", "hostile_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for auto labeling: {sorted(missing)}")

    result = df.copy()
    risk_score = pd.to_numeric(result["risk_score"], errors="coerce").fillna(0.0)
    profanity_count = pd.to_numeric(result["profanity_count"], errors="coerce").fillna(0)
    hostile_count = pd.to_numeric(result["hostile_count"], errors="coerce").fillna(0)

    risk_threshold = risk_score.quantile(0.8)
    bad_words_rule = (profanity_count > 0) | (hostile_count > 0)
    high_risk_rule = risk_score >= risk_threshold

    result["toxic"] = (bad_words_rule | high_risk_rule).astype(int)
    result["notes"] = result.get("notes", "").fillna("")
    result["notes"] = result["notes"].astype(str)
    result.loc[bad_words_rule, "notes"] = "auto: bad word / hostile word"
    result.loc[~bad_words_rule & high_risk_rule, "notes"] = f"auto: risk_score >= {risk_threshold:.3f}"
    result["label_source"] = f"auto_rules: bad_words or risk_score >= {risk_threshold:.3f}"

    if result["toxic"].nunique() < 2:
        raise ValueError("Auto labeling produced only one class.")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weak toxicity labels from dictionary/risk-score rules.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    labeled = auto_label(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Wrote: {args.output}")
    print(labeled["toxic"].value_counts().rename_axis("toxic").reset_index(name="count").to_string(index=False))


if __name__ == "__main__":
    main()
