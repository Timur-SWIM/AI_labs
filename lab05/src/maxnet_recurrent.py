"""Recurrent MAXNET with mutual inhibition."""
import numpy as np

def maxnet_recurrent(values, epsilon=0.1, max_steps=10000, tolerance=1e-12):
    """Iterate y_i <- max(0, y_i - epsilon * sum(other activations))."""
    y=np.asarray(values,dtype=float).reshape(-1).copy()
    if y.size == 0: raise ValueError("at least one potential is required")
    stable = epsilon > 0 and (y.size == 1 or epsilon < 1/(y.size-1))
    if not stable:
        raise ValueError("epsilon must be positive and stable: epsilon < 1/(n-1)")
    history=[y.copy()]
    for step in range(1,max_steps+1):
        total=y.sum(); new=np.maximum(0.0, y-epsilon*(total-y))
        history.append(new.copy()); y=new
        if np.count_nonzero(y > tolerance) <= 1:
            winner=int(np.argmax(y)); return winner,float(y[winner]),step,np.asarray(history)
    winner=int(np.argmax(y)); return winner,float(y[winner]),max_steps,np.asarray(history)

