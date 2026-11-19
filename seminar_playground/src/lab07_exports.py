"""Локальный ZIP-экспорт результатов ЛР7 для Orange."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _json(value):
    def default(item):
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Не сериализуемое значение: {type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, indent=2, default=default, allow_nan=False).encode("utf-8")


def orange_zip(*, dataset_manifest, train, test, control=None, predictions=None,
               control_predictions=None, experiment_config=None, metrics=None,
               report="# ЛР7\n", notes=None):
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise ValueError("Для экспорта нужны DataFrame Train и Test")
    files = {
        "dataset_manifest.json": _json(dataset_manifest),
        "train.csv": train.to_csv(index=False).encode("utf-8"),
        "test.csv": test.to_csv(index=False).encode("utf-8"),
        "report.md": report.encode("utf-8"),
    }
    if control is not None:
        files["control.csv"] = control.to_csv(index=False).encode("utf-8")
    if predictions is not None:
        files["predictions_test.csv"] = predictions.to_csv(index=False).encode("utf-8")
    if control_predictions is not None:
        files["control_predictions.csv"] = control_predictions.to_csv(index=False).encode("utf-8")
    if experiment_config is not None:
        files["experiment_config.json"] = _json(experiment_config)
    if metrics is not None:
        safe_metrics = dict(metrics)
        if not dataset_manifest.get("control_revealed", False):
            safe_metrics.pop("control", None)
        files["metrics.json"] = _json(safe_metrics)
    task = dataset_manifest.get("task", "не указан")
    readme = f"""# Orange — ЛР7

Пакет создан локально: {datetime.now(timezone.utc).isoformat()}.

- Режим: **{task}**.
- Источник: **{dataset_manifest.get('source', 'не указан')}**.
- `train.csv` используется для обучения, `test.csv` — для выбора модели.
- `control.csv` является независимой финальной оценкой и не должен подключаться к подбору.
- Признаки в CSV сохранены в исходном читаемом виде; нормализация Playground описана в manifest.

## Схема Orange

1. File → `train.csv` → Select Columns: назначьте Target, уберите ID/time из Features.
2. Select Columns → Neural Network → Test & Score (Data).
3. File → `test.csv` → Test & Score (Test Data), включите **Test on test data**.
4. Neural Network → Predictions (Model), `test.csv` → Predictions (Data) → Data Table.
5. Для classification добавьте Confusion Matrix и Scatter Plot; для regression — Scatter Plot.
6. Для forecasting используйте готовые `lag_1..lag_N` как Features и `target` как Continuous Target; это one-step pipeline.

Пути в File-виджетах после распаковки ZIP может потребоваться выбрать заново.

{notes or 'Файлы .ows не включены: Orange GUI в окружении разработки не установлен и не проверен.'}
"""
    files["README_ORANGE.md"] = readme.encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.writestr("orange_lab07_export/" + name, files[name])
    return output.getvalue()
