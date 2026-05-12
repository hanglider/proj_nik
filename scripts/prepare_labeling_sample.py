#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILES = (ROOT / "kpp.json", ROOT / "ls.json")
DEFAULT_OUTPUT = ROOT / "data" / "labeling_sample.csv"
DEFAULT_SUMMARY = ROOT / "data" / "dataset_summary.json"

PROFANITY_PATTERNS = [
    r"\bбля\w*",
    r"\bсука\w*",
    r"\bхуй\w*",
    r"\bхуя\w*",
    r"\bхуе\w*",
    r"\bпизд\w*",
    r"\bеба\w*",
    r"\bеби\w*",
    r"\bебл\w*",
    r"\bёба\w*",
    r"\bёби\w*",
    r"\bёб\w*",
    r"\bмуд\w*",
    r"\bдолбо\w*",
    r"\bгандон\w*",
]

HOSTILE_PATTERNS = [
    r"\bидиот\w*",
    r"\bтуп\w*",
    r"\bдебил\w*",
    r"\bурод\w*",
    r"\bзаткнись\b",
    r"\bненавиж\w*",
    r"\bсдох\w*",
    r"\bубью\b",
]

PROFANITY_RE = re.compile("|".join(PROFANITY_PATTERNS), flags=re.IGNORECASE)
HOSTILE_RE = re.compile("|".join(HOSTILE_PATTERNS), flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[\w_]{3,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
WHITESPACE_RE = re.compile(r"\s+")


def telegram_text_to_str(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def anonymize_text(text: str) -> str:
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = MENTION_RE.sub("<MENTION>", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def caps_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isupper() for ch in letters) / len(letters)


def message_features(text: str) -> dict[str, float | int]:
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    profanity_count = len(PROFANITY_RE.findall(text))
    hostile_count = len(HOSTILE_RE.findall(text))
    exclamation_count = text.count("!")
    question_count = text.count("?")
    caps = caps_ratio(text)
    all_caps_words = len(re.findall(r"\b[А-ЯA-ZЁ]{4,}\b", text))
    risk_score = (
        profanity_count * 3.0
        + hostile_count * 2.0
        + min(exclamation_count, 5) * 0.4
        + min(question_count, 5) * 0.2
        + min(caps * 3.0, 2.0)
        + all_caps_words * 0.8
    )
    return {
        "char_len": len(text),
        "word_count": len(words),
        "profanity_count": profanity_count,
        "hostile_count": hostile_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "caps_ratio": round(caps, 4),
        "risk_score": round(risk_score, 4),
    }


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def stable_hash(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_messages(paths: list[Path]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"files": {}, "total_raw_messages": 0, "total_kept_messages": 0}

    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        messages = payload.get("messages", [])
        file_summary = {
            "raw_messages": len(messages),
            "kept_messages": 0,
            "service_or_non_message": 0,
            "empty_text": 0,
        }
        chat_name = str(payload.get("name") or path.stem)

        for message in messages:
            if message.get("type") != "message":
                file_summary["service_or_non_message"] += 1
                continue

            text = anonymize_text(telegram_text_to_str(message.get("text", "")))
            if not text:
                file_summary["empty_text"] += 1
                continue

            date = parse_date(message.get("date"))
            raw_user_id = str(message.get("from_id") or "unknown")
            features = message_features(text)
            row = {
                "message_hash": stable_hash(path.name, message.get("id"), raw_user_id, text),
                "date": date.isoformat(sep=" ") if date else "",
                "hour": date.hour if date else "",
                "month": date.strftime("%Y-%m") if date else "",
                "chat": path.stem,
                "raw_user_id": raw_user_id,
                "text": text,
                **features,
            }
            rows.append(row)
            file_summary["kept_messages"] += 1

        summary["files"][path.name] = file_summary
        summary["total_raw_messages"] = int(summary["total_raw_messages"]) + file_summary["raw_messages"]
        summary["total_kept_messages"] = int(summary["total_kept_messages"]) + file_summary["kept_messages"]

    user_ids = sorted({str(row["raw_user_id"]) for row in rows})
    user_map = {user_id: f"user_{index:03d}" for index, user_id in enumerate(user_ids, start=1)}
    for row in rows:
        row["user"] = user_map[str(row.pop("raw_user_id"))]

    summary["anonymous_users"] = len(user_map)
    return rows, summary


def choose_sample(rows: list[dict[str, object]], sample_size: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    risky = [row for row in rows if float(row["risk_score"]) > 0]
    neutral_pool = [row for row in rows if float(row["risk_score"]) == 0]

    risky_size = min(len(risky), sample_size // 2)
    risky_sample = rng.sample(risky, risky_size) if risky_size else []
    risky_hashes = {row["message_hash"] for row in risky_sample}

    remaining = [row for row in rows if row["message_hash"] not in risky_hashes]
    random_size = min(sample_size - len(risky_sample), len(remaining))
    random_sample = rng.sample(remaining, random_size) if random_size else []

    sample: list[dict[str, object]] = []
    for row in risky_sample:
        sample.append({**row, "selection_reason": "risk_candidate"})
    for row in random_sample:
        sample.append({**row, "selection_reason": "random"})

    rng.shuffle(sample)
    for index, row in enumerate(sample, start=1):
        row["label_id"] = index
        row["toxic"] = ""
        row["notes"] = ""
    return sample


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label_id",
        "message_hash",
        "date",
        "hour",
        "month",
        "chat",
        "user",
        "text",
        "toxic",
        "notes",
        "selection_reason",
        "risk_score",
        "char_len",
        "word_count",
        "profanity_count",
        "hostile_count",
        "exclamation_count",
        "question_count",
        "caps_ratio",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare anonymized Telegram messages for manual toxicity labeling.")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("data_files", nargs="*", type=Path, default=list(DEFAULT_DATA_FILES))
    args = parser.parse_args()

    missing = [path for path in args.data_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files: {', '.join(str(path) for path in missing)}")

    rows, summary = load_messages(args.data_files)
    sample = choose_sample(rows, sample_size=args.sample_size, seed=args.seed)
    write_csv(sample, args.output)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {**summary, "labeling_sample_size": len(sample), "labeling_sample_path": str(args.output)}
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Kept messages: {summary['total_kept_messages']}")
    print(f"Anonymous users: {summary['anonymous_users']}")
    print(f"Labeling sample: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
