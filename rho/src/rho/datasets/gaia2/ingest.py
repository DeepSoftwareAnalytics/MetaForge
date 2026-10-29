from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs


SUPPORTED_CONFIGS = (
    "mini",
    "execution",
    "search",
    "adaptability",
    "time",
    "ambiguity",
    "demo",
)
DEFAULT_CONFIG = "mini"


@dataclass(frozen=True)
class Gaia2Payload:
    dataset_spec: str
    config: str = DEFAULT_CONFIG


@dataclass(frozen=True)
class Gaia2Row:
    id: str
    scenario_id: str
    config: str
    data: dict[str, Any]

    @property
    def task_id(self) -> str:
        return f"{self.config}/{self.scenario_id}"


def parse_payload(payload: str) -> Gaia2Payload:
    dataset_spec, sep, fragment = payload.partition("#")
    if not dataset_spec:
        raise ValueError("GAIA-2 dataset spec must not be empty")
    config = DEFAULT_CONFIG
    if sep:
        params = parse_qs(fragment, keep_blank_values=True)
        unknown = sorted(set(params) - {"config"})
        if unknown:
            raise ValueError(f"Unsupported GAIA-2 payload option(s): {', '.join(unknown)}")
        values = params.get("config")
        if values:
            config = values[-1]
    if config not in SUPPORTED_CONFIGS:
        allowed = ", ".join(SUPPORTED_CONFIGS)
        raise ValueError(f"Unsupported GAIA-2 config {config!r}; expected one of: {allowed}")
    return Gaia2Payload(dataset_spec=dataset_spec, config=config)


def load_rows(payload: Gaia2Payload) -> tuple[Gaia2Row, ...]:
    path = Path(payload.dataset_spec).expanduser()
    if path.exists():
        return _load_local_rows(path, config=payload.config)
    return _load_hf_rows(payload)


def _load_local_rows(path: Path, *, config: str) -> tuple[Gaia2Row, ...]:
    if path.is_dir():
        candidates = [
            path / f"{config}.jsonl",
            path / f"{config}.json",
            path / "data.jsonl",
            path / "data.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No GAIA-2 local data file found under {path}; expected {config}.jsonl or data.jsonl"
            )
    rows: Iterable[dict[str, Any]]
    if path.suffix == ".jsonl":
        parsed_rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parsed_rows.append(json.loads(line))
        rows = parsed_rows
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        else:
            raise ValueError(f"Unsupported GAIA-2 local JSON shape in {path}")
    return tuple(_coerce_row(row, default_config=config) for row in rows)


def _load_hf_rows(payload: Gaia2Payload) -> tuple[Gaia2Row, ...]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading GAIA-2 from HuggingFace requires the `datasets` package. "
            "Install rho with the `gaia2` extra."
        ) from exc

    dataset = load_dataset(payload.dataset_spec, payload.config, split="validation")
    return tuple(_coerce_row(dict(row), default_config=payload.config) for row in dataset)


def _coerce_row(row: dict[str, Any], *, default_config: str) -> Gaia2Row:
    scenario_id = str(row.get("scenario_id") or row.get("id") or "")
    if not scenario_id:
        raise ValueError("GAIA-2 row is missing scenario_id/id")
    config = str(row.get("config") or row.get("split") or default_config)
    data = row.get("data", {})
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError(f"GAIA-2 row {scenario_id!r} data must be an object or JSON object string")
    row_id = str(row.get("id") or scenario_id)
    return Gaia2Row(id=row_id, scenario_id=scenario_id, config=config, data=data)
