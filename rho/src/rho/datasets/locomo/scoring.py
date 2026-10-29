"""Byte-for-byte port of LOCOMO's ``task_eval/evaluation.py`` scoring.

The upstream has non-documented quirks that materially affect the F1
numbers (category 1 splits on commas, category 3 silently discards
alternatives after the first semicolon, stopword list is
``{a, an, the, and}`` — not NLTK's default). Preserving these lets us
report numbers directly comparable to Mem0 / Zep / MIRIX / LightMem.

See ``docs/superpowers/specs/2026-04-11-locomo-dataset-design.md`` §9
for the full rules.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from nltk.stem.porter import PorterStemmer

_ps = PorterStemmer()


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = " ".join(s.split())
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    per_gold = [
        max(f1_score(pred, gt) for pred in predictions) for gt in ground_truths
    ]
    return sum(per_gold) / len(per_gold) if per_gold else 0.0


def score_qa(prediction: str, gold: str, category: int) -> float:
    """Score one LOCOMO QA by category, matching upstream eval_question_answering."""
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category in (2, 3, 4):
        return float(f1_score(prediction, gold))
    if category == 1:
        return float(f1_multi(prediction, gold))
    raise ValueError(f"Unsupported LOCOMO category: {category}")


_ANSWER_SENTINEL = re.compile(r"^ANSWER:[ \t]*(.*)$", re.MULTILINE)


def extract_answer(final_message: str) -> str:
    """Extract the substring after the last ``ANSWER:`` sentinel.

    Handles both ``ANSWER: foo`` (inline) and ``ANSWER:\\nfoo`` (next
    line). If no sentinel is found, return the full message — better
    to give the scorer a noisy prediction than to discard partial credit.
    """
    matches = list(_ANSWER_SENTINEL.finditer(final_message))
    if not matches:
        return final_message
    last = matches[-1]
    inline = last.group(1).strip()
    if inline:
        return inline
    tail = final_message[last.end() :]
    for line in tail.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
