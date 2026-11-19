"""Безопасный импорт и проверка таблиц ЛР7."""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 128
_ALLOWED_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")


@dataclass(frozen=True)
class UploadedTable:
    name: str
    frame: pd.DataFrame
    size_bytes: int
    sha256: str
    encoding: str
    delimiter: str


def decode_bytes(data: bytes, encoding: str = "auto") -> tuple[str, str]:
    if len(data) > MAX_BYTES:
        raise ValueError("Файл превышает ограничение 10 MiB")
    if not data or b"\x00" in data:
        raise ValueError("Файл пуст или похож на бинарный")
    encodings = _ALLOWED_ENCODINGS if encoding == "auto" else (encoding,)
    for enc in encodings:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("Не удалось декодировать файл; попробуйте UTF-8, UTF-8-BOM или Windows-1251")


def detect_delimiter(text: str) -> str:
    lines = [line for line in text.splitlines()[:20] if line.strip()]
    candidates = [",", ";", "\t"]
    scored = []
    for delimiter in candidates:
        counts = [line.count(delimiter) for line in lines]
        positive = [count for count in counts if count]
        # Разделитель таблицы обычно встречается стабильно по строкам; запятая
        # внутри 1,25 даёт нестабильный профиль и проигрывает ';'.
        consistency = len(positive) == len(counts) and len(set(positive)) == 1
        scored.append((consistency, sum(counts), delimiter))
    consistent = [item for item in scored if item[0] and item[1] > 0]
    if consistent:
        return max(consistent, key=lambda item: item[1])[2]
    best = max(scored, key=lambda item: item[1])
    return best[2] if best[1] else r"\s+"


def parse_uploaded(data: bytes, *, encoding: str = "auto", delimiter: str | None = None,
                   decimal: str = "auto", header: str | int | None = "infer") -> pd.DataFrame:
    text, _ = decode_bytes(data, encoding)
    sep = delimiter or detect_delimiter(text)
    if decimal == "auto":
        decimal = "," if sep == ";" and re.search(r"\d,\d", text) else "."
    if sep not in {",", ";", "\t", r"\s+"}:
        raise ValueError("Разделитель должен быть comma, semicolon, tab или whitespace")
    if decimal not in {".", ","}:
        raise ValueError("Десятичный разделитель должен быть точкой или запятой")
    try:
        frame = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, header=header, engine="python")
    except Exception as exc:
        raise ValueError(f"Не удалось разобрать таблицу: {exc}") from exc
    if frame.empty:
        raise ValueError("Таблица не содержит строк")
    if frame.shape[0] > MAX_ROWS or frame.shape[1] > MAX_COLUMNS:
        raise ValueError(f"Слишком много данных: максимум {MAX_ROWS} строк и {MAX_COLUMNS} столбцов")
    frame.columns = [str(column).strip() for column in frame.columns]
    normalized = [re.sub(r"\.\d+$", "", column) for column in frame.columns]
    if any(not column for column in frame.columns) or len(set(normalized)) != len(normalized):
        raise ValueError("Имена столбцов должны быть непустыми и уникальными")
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Столбец {column!r} содержит NaN или Inf")
    return frame


def parse_file(name: str, data: bytes, **kwargs) -> UploadedTable:
    text, used_encoding = decode_bytes(data, kwargs.get("encoding", "auto"))
    frame = parse_uploaded(data, **kwargs)
    return UploadedTable(name, frame, len(data), hashlib.sha256(data).hexdigest(), used_encoding,
                         kwargs.get("delimiter") or detect_delimiter(text))


def file_info(name: str, data: bytes, frame: pd.DataFrame) -> dict:
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(frame), "columns": len(frame.columns)}


def suggest_task(frame: pd.DataFrame) -> str:
    """Эвристика только для подсказки, а не окончательное решение."""
    names = " ".join(map(str, frame.columns))
    if re.search(r"time|date|timestamp|index|время|дата", names, re.I):
        return "timeseries"
    if frame.shape[1] >= 2 and any(frame[column].nunique(dropna=True) <= 10 for column in frame.columns):
        return "classification"
    return "regression"


def validate_schema(train: pd.DataFrame, test: pd.DataFrame | None = None,
                    control: pd.DataFrame | None = None, features: list[str] | None = None,
                    target: str | None = None) -> bool:
    if train is None or train.empty:
        raise ValueError("Train пуст")
    features = list(features or [])
    if not features:
        raise ValueError("Выберите хотя бы один признак")
    required = features + ([target] if target else [])
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError("Не найдены столбцы: " + ", ".join(missing))
    if len(set(features)) != len(features):
        raise ValueError("Признаки не должны повторяться")
    for name, part in (("Test", test), ("Control", control)):
        if part is not None and list(part.columns) != list(train.columns):
            raise ValueError(f"{name}: имена и порядок столбцов должны совпадать с Train")
    return True
