"""Safe in-memory export builders."""
from __future__ import annotations
import io, json
import pandas as pd
import matplotlib.pyplot as plt

def csv_bytes(frame: pd.DataFrame): return frame.to_csv(index=False).encode("utf-8")
def json_bytes(value): return json.dumps(value, ensure_ascii=False, indent=2, default=float).encode("utf-8")
def markdown_report(record):
    return ("# MLP Playground — результат\n\n" + "| Параметр | Значение |\n|---|---:|\n" +
            "\n".join(f"| {k} | {v} |" for k,v in record.items() if k != "history") + "\n").encode("utf-8")

def parity_png(y_true, y_pred):
    """Render the currently computed prediction, not an archived plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true[:, 0], y_pred[:, 0], label="x")
    ax.scatter(y_true[:, 1], y_pred[:, 1], label="y")
    ax.plot([0, 1], [0, 1], "k--", label="ŷ=y")
    ax.set(xlabel="true", ylabel="predicted", title="True vs Predicted")
    ax.legend(); fig.tight_layout(); out = io.BytesIO(); fig.savefig(out, format="png", dpi=150); plt.close(fig)
    return out.getvalue()
