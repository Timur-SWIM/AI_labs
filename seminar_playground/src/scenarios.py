"""Reproducible diagnosis settings. Symptoms are computed, never drawn."""
from __future__ import annotations

SCENARIOS = {
    "A — слишком медленно": {"layers": (2,7,3,2), "optimizer": "adam", "learning_rate": .0001, "epochs": 250, "init_scale": 1., "seed": 42, "goal": "Уменьшите validation MSE хотя бы на 25% за 250 эпох."},
    "B — неустойчиво": {"layers": (2,7,3,2), "optimizer": "adam", "learning_rate": 1.0, "epochs": 120, "init_scale": 1., "seed": 42, "goal": "Добейтесь конечной validation MSE < 0.02 без NaN/Inf."},
    "C — обобщение": {"layers": (2,16,16,2), "optimizer": "adam", "learning_rate": .02, "epochs": 1000, "init_scale": 1., "seed": 19, "goal": "Снизьте разрыв validation/train и validation MSE; объясните компромисс."},
}

def success(name, before, after):
    if not after or not after.get("validation"): return False, "Сначала выполните обучение."
    val = after["validation"]["mse"]
    if not __import__("numpy").isfinite(val): return False, "Есть нечисловая ошибка: режим неустойчив."
    old = before.get("validation", {}).get("mse", float("inf")) if before else float("inf")
    if name.startswith("A"): return val <= old*.75, "Нужно улучшить validation MSE минимум на 25%."
    if name.startswith("B"): return val < .02, "Цель: validation MSE < 0.02 и конечные значения."
    gap = abs(after["validation"]["mse"]-after["train"]["mse"])
    return val < old and gap < max(abs(old-before["train"]["mse"]), .01), "Нужно улучшить validation и уменьшить разрыв train/validation."
