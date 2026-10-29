"""Feed-forward MAXNET implemented with explicit pairwise comparators."""
import numpy as np

def comparator(a, b):
    """Two-input winner-take-all comparator."""
    return a if a >= b else b

def maxnet_feedforward(values):
    """Return (winner index, winning value, comparison trace).

    A sequential comparator tournament is used instead of calling argmax.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one potential is required")
    winner, best = 0, float(values[0]); trace=[]
    for idx in range(1, values.size):
        old = best; best = comparator(best, float(values[idx]))
        if best != old:
            winner = idx
        trace.append({"left_index": winner if best == old else idx, "candidate_index": idx, "winner_value": best})
    return winner, best, trace

