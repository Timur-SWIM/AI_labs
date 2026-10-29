from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).parents[1]


def test_data_and_splits():
    assert len(pd.read_csv(ROOT / "data/coord.csv")) == 100
    assert len(pd.read_csv(ROOT / "data/coord_train.csv")) == 60
    assert len(pd.read_csv(ROOT / "data/coord_test.csv")) == 20
    assert len(pd.read_csv(ROOT / "data/coord_control.csv")) == 20
    assert list(pd.read_csv(ROOT / "data/coord_x.csv").columns) == ["rho", "phi", "x"]
    assert list(pd.read_csv(ROOT / "data/coord_y.csv").columns) == ["rho", "phi", "y"]


def test_all_experiments_exist():
    expected = ["base_history.csv", "base_metrics.json", "lr_sweep.csv",
                "bad_regime_histories.csv", "stop80_metrics.json",
                "architecture_comparison.csv", "predictions.csv", "summary.json"]
    assert all((ROOT / "results" / name).stat().st_size > 0 for name in expected)
    assert len(pd.read_csv(ROOT / "results/lr_sweep.csv")) == 6
    assert len(pd.read_csv(ROOT / "results/architecture_comparison.csv")) == 5


def test_acceptance_metrics():
    summary = json.loads((ROOT / "results/summary.json").read_text())
    assert summary["base"]["architecture"] == "2-7-3-2"
    assert summary["stop80"]["train"]["gf_fraction"] >= .8
    assert summary["chosen"]["test_mse"] < .01

