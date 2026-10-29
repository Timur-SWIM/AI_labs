from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).parents[1]

def test_outputs_exist_and_are_nonempty():
    for name in ["xor_predictions.csv","xor_metrics.json","approx_predictions.csv","approx_metrics.csv","approx_weights.json","classification_metrics.json","iris_metrics.json"]:
        p=ROOT/"results"/name
        assert p.exists() and p.stat().st_size > 0
    for name in ["xor_training_curve.png","approx_true_vs_pred.png","approx_training_curve.png","approx_residuals.png","classification_decision_regions.png","classification_confusion_matrix.png","iris_scatter.png","iris_confusion_matrix.png"]:
        assert (ROOT/"plots"/name).exists()

def test_xor_is_solved():
    metrics=json.loads((ROOT/"results/xor_metrics.json").read_text())
    assert max(v["accuracy"] for v in metrics.values()) == 1.0

def test_orange_csv_shapes():
    assert list(pd.read_csv(ROOT/"data/xor.csv").columns)==["x1","x2","result"]
    classification = pd.read_csv(ROOT/"data/classification.csv")
    # `class` is a required Orange target name, but also a Python keyword;
    # use explicit column indexing instead of DataFrame.query().
    assert (classification["class"] == "A").sum() >= 10
    assert (classification["class"] == "B").sum() >= 10

