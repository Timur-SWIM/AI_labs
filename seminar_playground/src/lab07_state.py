"""Идентичность эксперимента и безопасная привязка контрольных результатов."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def frame_digest(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_fingerprint(*, parts: dict[str, pd.DataFrame | None], task: str, features: list[str],
                        target: str | None, time_column: str | None = None, value_column: str | None = None,
                        lag: int | None = None, preprocessing: dict | None = None,
                        model: dict | None = None) -> str:
    payload = {
        "task": task, "features": list(features), "target": target,
        "time_column": time_column, "value_column": value_column, "lag": lag,
        "preprocessing_version": "train-only-v2", "preprocessing": preprocessing or {},
        "parts": {name: None if frame is None else {"rows": len(frame), "columns": list(map(str, frame.columns)), "sha256": frame_digest(frame)} for name, frame in sorted(parts.items())},
        "model": model or {},
    }
    canonical = json.dumps(_json_value(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def short_fingerprint(value: str) -> str:
    return value[:12] if value else "—"
