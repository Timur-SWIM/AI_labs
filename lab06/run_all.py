#!/usr/bin/env python3
"""Run all Lab 06 coordinate-transform MLP experiments."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.mlp import MLPRegressor

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, PLOTS, REPORT = (ROOT / name for name in ("data", "results", "plots", "report"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def savefig(path, dpi):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def load_or_generate(seed, count):
    candidates = [ROOT / "coord.dat", DATA / "coord.dat", ROOT.parent / "coord.dat"]
    source = next((p for p in candidates if p.exists()), None)
    if source:
        raw = np.loadtxt(source, delimiter=None)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("coord.dat must contain at least rho and phi columns")
        rho, phi = raw[:, 0], raw[:, 1]
        if raw.shape[1] >= 4:
            x, y = raw[:, 2], raw[:, 3]
        else:
            x, y = rho * np.cos(phi), rho * np.sin(phi)
        generated = False
    else:
        rng = np.random.default_rng(seed)
        rho = rng.uniform(0.05, 1.0, count)
        phi = rng.uniform(0.0, np.pi / 2, count)
        x, y = rho * np.cos(phi), rho * np.sin(phi)
        generated = True
    frame = pd.DataFrame({"rho": rho, "phi": phi, "x": x, "y": y})
    frame.to_csv(DATA / "coord.csv", index=False)
    frame[["rho", "phi", "x"]].to_csv(DATA / "coord_x.csv", index=False)
    frame[["rho", "phi", "y"]].to_csv(DATA / "coord_y.csv", index=False)
    return frame, generated, str(source) if source else None


def split_data(frame, seed, sizes):
    if len(frame) < sum(sizes):
        raise ValueError(f"need at least {sum(sizes)} coordinate rows")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))[:sum(sizes)]
    names, chunks, start = ("train", "test", "control"), {}, 0
    for name, size in zip(names, sizes):
        part = frame.iloc[order[start:start + size]].copy()
        part.insert(0, "sample_id", part.index)
        part.to_csv(DATA / f"coord_{name}.csv", index=False)
        chunks[name] = part
        start += size
    return chunks


def arrays(frame, mean, scale):
    x = (frame[["rho", "phi"]].to_numpy(float) - mean) / scale
    y = frame[["x", "y"]].to_numpy(float)
    return x, y


def metrics(model, x, y, tolerance):
    pred = model.predict(x)
    abs_error = np.abs(y - pred)
    gf = int(np.all(abs_error <= tolerance, axis=1).sum())
    return {"mse": model.mse(y, pred), "mae": float(abs_error.mean()),
            "gf": gf, "bf": int(len(y) - gf), "gf_fraction": gf / len(y)}


def history_frame(model, experiment, **extra):
    result = pd.DataFrame(model.history)
    result.insert(0, "experiment", experiment)
    for key, value in extra.items():
        result[key] = value
    return result


def train_model(layers, seed, x_train, y_train, x_test, y_test, *, lr=.01,
                epochs=3000, init_scale=1.0, optimizer="adam", patience=None,
                tolerance=.05, stop_gf=None):
    model = MLPRegressor(layers, seed=seed, init_scale=init_scale)
    model.fit(x_train, y_train, x_test, y_test, learning_rate=lr,
              max_epochs=epochs, optimizer=optimizer, patience=patience,
              tolerance=tolerance, stop_good_fraction=stop_gf)
    return model


def render_report(summary):
    base, chosen, stop = summary["base"], summary["chosen"], summary["stop80"]
    text = f"""# Лабораторная работа №6

## Цель и данные

Исследован MLP с backpropagation для преобразования `(ρ, φ) → (x, y)`, где
`x=ρ cosφ`, `y=ρ sinφ`. Файл `coord.dat` не найден, поэтому с seed 42 создано
100 точек первого квадранта: `ρ∈[0.05,1]`, `φ∈[0,π/2]`. Разбиение:
60 train, 20 test, 20 control. Для Orange сохранены отдельные задачи
`coord_x.csv` и `coord_y.csv`.

Порог GF принят равным 0.05: пример считается хорошим, если одновременно
`|x−x̂|≤0.05` и `|y−ŷ|≤0.05`. Это допущение, так как порог варианта не дан.

## Базовая сеть 2–7–3–2

Сеть использует tanh в скрытых слоях, линейный выход, MSE и Adam. Обучение
завершилось на эпохе {base['epochs']}. Фактические показатели:

| Набор | MSE | MAE | GF | BF |
|---|---:|---:|---:|---:|
| train | {base['train']['mse']:.8f} | {base['train']['mae']:.8f} | {base['train']['gf']} | {base['train']['bf']} |
| test | {base['test']['mse']:.8f} | {base['test']['mae']:.8f} | {base['test']['gf']} | {base['test']['bf']} |
| control | {base['control']['mse']:.8f} | {base['control']['mae']:.8f} | {base['control']['gf']} | {base['control']['bf']} |

## Эксперименты

Выполнен sweep learning rate `0.0005, 0.001, 0.005, 0.01, 0.05, 0.1`.
Истории находятся в `lr_histories.csv`, итог — в `lr_sweep.csv`. Насыщение
воспроизведено крупной инициализацией tanh (`init_scale=15`) и медленным SGD;
слишком высокий шаг — Adam с learning rate 1.0. Их реальные кривые записаны
в `bad_regime_histories.csv`.

При остановке по первому достижению 80% GF обучение остановилось на эпохе
{stop['epochs']} с train GF={stop['train']['gf_fraction']:.3f}; на test получено
GF={stop['test']['gf']}, BF={stop['test']['bf']}, MSE={stop['test']['mse']:.8f}.

Сравнены пять архитектур. По минимальному test MSE выбрана **{chosen['architecture']}**:
{chosen['parameters']} параметров, test MSE={chosen['test_mse']:.8f},
test GF={chosen['test_gf']}, BF={chosen['test_bf']}. Выбор сделан по
генерализации на test, а не по train loss. Независимая оценка выбранной сети
на control: MSE={chosen['control_mse']:.8f}, GF={chosen['control_gf']},
BF={chosen['control_bf']}.

## Вывод

Backpropagation успешно аппроксимирует нелинейное преобразование координат.
Слишком малая скорость замедляет обучение, чрезмерная ухудшает устойчивость,
а насыщение tanh уменьшает градиенты. Увеличение сети не гарантирует лучшую
генерализацию, поэтому архитектура выбрана по test error и затем проверена на
не использованном при выборе control-наборе.
"""
    (REPORT / "report.md").write_text(text, encoding="utf-8")


def main():
    for directory in (DATA, RESULTS, PLOTS, REPORT):
        directory.mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed, tolerance, dpi = cfg["random_seed"], cfg["tolerance"], 180
    plt.rcParams.update({"figure.dpi": dpi, "savefig.dpi": dpi, "axes.grid": True})
    frame, generated, source = load_or_generate(seed, cfg["sample_count"])
    parts = split_data(frame, seed, [cfg["split"][k] for k in ("train", "test", "control")])
    input_mean = parts["train"][["rho", "phi"]].to_numpy().mean(axis=0)
    input_scale = parts["train"][["rho", "phi"]].to_numpy().std(axis=0)
    xtr, ytr = arrays(parts["train"], input_mean, input_scale)
    xte, yte = arrays(parts["test"], input_mean, input_scale)
    xco, yco = arrays(parts["control"], input_mean, input_scale)

    base_layers = cfg["base_architecture"]
    base = train_model(base_layers, seed, xtr, ytr, xte, yte,
                       lr=cfg["base_learning_rate"], epochs=cfg["max_epochs"],
                       tolerance=tolerance, patience=500)
    base_history = history_frame(base, "base")
    base_history.to_csv(RESULTS / "base_history.csv", index=False)
    base_metrics = {name: metrics(base, x, y, tolerance) for name, x, y in
                    [("train", xtr, ytr), ("test", xte, yte), ("control", xco, yco)]}
    base_metrics.update({"architecture": "-".join(map(str, base_layers)),
                         "parameters": base.parameter_count, "epochs": len(base.history)})
    save_json(RESULTS / "base_metrics.json", base_metrics)
    save_json(RESULTS / "base_weights.json", {"weights": [w.tolist() for w in base.weights],
                                               "biases": [b.tolist() for b in base.biases]})

    plt.figure()
    plt.semilogy(base_history.epoch, base_history.train_loss, label="train")
    plt.semilogy(base_history.epoch, base_history.validation_loss, label="validation/test")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Base 2–7–3–2 loss"); plt.legend()
    savefig(PLOTS / "base_loss_curves.png", dpi)

    lr_rows, lr_histories = [], []
    for lr in cfg["learning_rates"]:
        model = train_model(base_layers, seed, xtr, ytr, xte, yte, lr=lr,
                            epochs=1500, tolerance=tolerance)
        mt, mv = metrics(model, xtr, ytr, tolerance), metrics(model, xte, yte, tolerance)
        lr_rows.append({"learning_rate": lr, "epochs": len(model.history),
                        "train_mse": mt["mse"], "test_mse": mv["mse"],
                        "test_gf": mv["gf"], "test_bf": mv["bf"]})
        lr_histories.append(history_frame(model, f"lr={lr}", learning_rate=lr))
    pd.DataFrame(lr_rows).to_csv(RESULTS / "lr_sweep.csv", index=False)
    lr_history = pd.concat(lr_histories, ignore_index=True)
    lr_history.to_csv(RESULTS / "lr_histories.csv", index=False)
    plt.figure()
    for lr, group in lr_history.groupby("learning_rate", sort=False):
        plt.semilogy(group.epoch, group.validation_loss, label=str(lr))
    plt.xlabel("Epoch"); plt.ylabel("Validation MSE"); plt.title("Learning-rate comparison")
    plt.legend(title="learning rate", ncol=2); savefig(PLOTS / "learning_rate_comparison.png", dpi)

    paralysis = train_model(base_layers, seed, xtr, ytr, xte, yte, lr=.001,
                            epochs=1000, init_scale=15, optimizer="sgd", tolerance=tolerance)
    high_lr = train_model(base_layers, seed, xtr, ytr, xte, yte, lr=1.0,
                          epochs=500, optimizer="adam", tolerance=tolerance)
    bad_history = pd.concat([history_frame(paralysis, "saturation_paralysis"),
                             history_frame(high_lr, "high_learning_rate")], ignore_index=True)
    bad_history.to_csv(RESULTS / "bad_regime_histories.csv", index=False)
    for model, name, title in [(paralysis, "paralysis_case.png", "Saturation / paralysis"),
                               (high_lr, "high_lr_case.png", "Excessive learning rate")]:
        hist = pd.DataFrame(model.history); plt.figure(); plt.semilogy(hist.epoch, hist.train_loss, label="train")
        plt.semilogy(hist.epoch, hist.validation_loss, label="validation/test"); plt.xlabel("Epoch")
        plt.ylabel("MSE"); plt.title(title); plt.legend(); savefig(PLOTS / name, dpi)

    stop_model = train_model(base_layers, seed, xtr, ytr, xte, yte, lr=.01,
                             epochs=cfg["max_epochs"], tolerance=tolerance, stop_gf=.8)
    history_frame(stop_model, "stop_at_80_percent").to_csv(RESULTS / "stop80_history.csv", index=False)
    stop_metrics = {name: metrics(stop_model, x, y, tolerance) for name, x, y in
                    [("train", xtr, ytr), ("test", xte, yte), ("control", xco, yco)]}
    stop_metrics["epochs"] = len(stop_model.history)
    save_json(RESULTS / "stop80_metrics.json", stop_metrics)

    architecture_rows, models = [], {}
    for architecture in cfg["architectures"]:
        model = train_model(architecture, seed, xtr, ytr, xte, yte, lr=.01,
                            epochs=cfg["max_epochs"], tolerance=tolerance, patience=500)
        train_m, test_m, control_m = (metrics(model, *pair, tolerance) for pair in
                                      [(xtr, ytr), (xte, yte), (xco, yco)])
        name = "-".join(map(str, architecture)); models[name] = model
        architecture_rows.append({"architecture": name, "parameters": model.parameter_count,
                                  "epochs": len(model.history), "train_mse": train_m["mse"],
                                  "test_mse": test_m["mse"], "control_mse": control_m["mse"],
                                  "train_gf": train_m["gf"], "train_bf": train_m["bf"],
                                  "test_gf": test_m["gf"], "test_bf": test_m["bf"],
                                  "control_gf": control_m["gf"], "control_bf": control_m["bf"]})
    architecture_df = pd.DataFrame(architecture_rows).sort_values("test_mse").reset_index(drop=True)
    architecture_df.to_csv(RESULTS / "architecture_comparison.csv", index=False)
    chosen_row = architecture_df.iloc[0].to_dict(); chosen_name = chosen_row["architecture"]
    chosen = models[chosen_name]
    save_json(RESULTS / "selected_model.json", {"selection_rule": "minimum test MSE",
                                                 "architecture": chosen_name,
                                                 "metrics": chosen_row,
                                                 "weights": [w.tolist() for w in chosen.weights],
                                                 "biases": [b.tolist() for b in chosen.biases]})

    predictions = []
    for split, part, xx, yy in [("train", parts["train"], xtr, ytr),
                                ("test", parts["test"], xte, yte),
                                ("control", parts["control"], xco, yco)]:
        pred = chosen.predict(xx)
        for sid, true, estimated in zip(part.sample_id, yy, pred):
            predictions.append({"split": split, "sample_id": int(sid), "x_true": true[0],
                                "x_pred": estimated[0], "y_true": true[1], "y_pred": estimated[1],
                                "abs_error_x": abs(true[0] - estimated[0]),
                                "abs_error_y": abs(true[1] - estimated[1]),
                                "gf": bool(np.all(np.abs(true - estimated) <= tolerance))})
    pred_df = pd.DataFrame(predictions); pred_df.to_csv(RESULTS / "predictions.csv", index=False)
    test_pred = pred_df[pred_df.split == "test"].sort_values("x_true")
    for coordinate in ("x", "y"):
        ordered = test_pred.sort_values(f"{coordinate}_true")
        plt.figure(); plt.plot(range(len(ordered)), ordered[f"{coordinate}_true"], "o-", label="true")
        plt.plot(range(len(ordered)), ordered[f"{coordinate}_pred"], "s--", label="predicted")
        plt.xlabel("Test sample (sorted)"); plt.ylabel(coordinate); plt.title(f"{coordinate}: true vs predicted")
        plt.legend(); savefig(PLOTS / f"{coordinate}_true_pred.png", dpi)
    plt.figure(); plt.hist([test_pred.abs_error_x, test_pred.abs_error_y], bins=10, label=["|dx|", "|dy|"], alpha=.75)
    plt.axvline(tolerance, color="black", linestyle="--", label="GF tolerance")
    plt.xlabel("Absolute error"); plt.ylabel("Count"); plt.title("Test error distribution"); plt.legend()
    savefig(PLOTS / "error_distribution.png", dpi)

    summary = {"data": {"generated": generated, "source": source, "samples": len(frame)},
               "tolerance": tolerance, "base": base_metrics, "stop80": stop_metrics,
               "chosen": chosen_row}
    save_json(RESULTS / "summary.json", summary)
    render_report(summary)
    print(f"Lab 06 completed; selected architecture: {chosen_name}")


if __name__ == "__main__":
    main()

