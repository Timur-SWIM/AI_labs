#!/usr/bin/env python3
"""Воспроизводимый demo smoke-pipeline ЛР7; не подменяет данные варианта."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    root = Path(__file__).resolve().parent
    playground = root.parent / "seminar_playground"
    sys.path.insert(0, str(playground))
    from src.lab07_datasets import regression_demo
    from src.lab07_engine import Lab07Engine
    from src.lab07_metrics import regression_metrics

    data_dir, result_dir, plot_dir, report_dir = [root / name for name in ("data", "results", "plots", "report")]
    for directory in (data_dir, result_dir, plot_dir, report_dir): directory.mkdir(exist_ok=True)
    dataset = regression_demo(42, 180)
    frame = dataset.frame
    rng = np.random.default_rng(42); order = rng.permutation(len(frame)); a, b = 108, 144
    parts = {"train": frame.iloc[order[:a]].copy(), "test": frame.iloc[order[a:b]].copy(), "control": frame.iloc[order[b:]].copy()}
    for name, part in parts.items(): part.to_csv(data_dir / f"{name}.csv", index=False)
    mean = parts["train"][["x1", "x2"]].to_numpy().mean(0); scale = parts["train"][["x1", "x2"]].to_numpy().std(0); scale[scale == 0] = 1
    arrays = lambda part: ((part[["x1", "x2"]].to_numpy() - mean) / scale, part[["target"]].to_numpy())
    x_train, y_train = arrays(parts["train"]); x_test, y_test = arrays(parts["test"])
    rows = []
    for architecture in ((2, 2, 1), (2, 4, 1), (2, 8, 1), (2, 8, 4, 1), (2, 16, 8, 1), (2, 16, 16, 1)):
        engine = Lab07Engine(architecture, seed=42, learning_rate=.01)
        engine.train_epochs(300, x_train, y_train, x_test, y_test)
        train_metrics = regression_metrics(y_train, engine.predict(x_train)); test_metrics = regression_metrics(y_test, engine.predict(x_test))
        rows.append({"architecture": "-".join(map(str, architecture)), "parameters": engine.parameter_count, "epochs": engine.epoch, "train_mse": train_metrics["mse"], "test_mse": test_metrics["mse"], "train_gf": train_metrics["gf"], "train_bf": train_metrics["bf"], "test_gf": test_metrics["gf"], "test_bf": test_metrics["bf"], "test_max_error": test_metrics["max_error"], "test_mean_error": test_metrics["mean_error"]})
    table = pd.DataFrame(rows).sort_values("test_mse"); table.to_csv(result_dir / "architecture_comparison.csv", index=False)
    best = table.iloc[0].to_dict(); (result_dir / "summary.json").write_text(json.dumps({"source": "demo", "formula": dataset.formula, "seed": 42, "selected_by": "test_mse", "selected": best}, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.figure(); plt.scatter(frame.x1, frame.target, s=12); plt.xlabel("x1"); plt.ylabel("target"); plt.title("Demo regression data"); plt.tight_layout(); plt.savefig(plot_dir / "regression_data.png", dpi=160); plt.close()
    (report_dir / "report.md").write_text("# ЛР7 demo smoke-test\n\nВариантные файлы отсутствуют; этот отчёт не является результатом индивидуального варианта. Фактическое сравнение шести архитектур сохранено в `results/architecture_comparison.csv`.\n", encoding="utf-8")
    print(f"Demo smoke-test completed; selected architecture: {best['architecture']}")

if __name__ == "__main__": main()
