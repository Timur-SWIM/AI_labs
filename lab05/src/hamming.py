"""Hamming layer for bipolar binary patterns."""
import numpy as np

def bipolarize(pattern):
    """Return a flat bipolar vector with values -1 and +1."""
    a = np.asarray(pattern)
    return np.where(a > 0, 1.0, -1.0).reshape(-1)

def hamming_distance(a, b):
    a, b = bipolarize(a), bipolarize(b)
    if a.size != b.size:
        raise ValueError("patterns must have equal size")
    return int(np.count_nonzero(a != b))

def similarity(a, b):
    """Normalized agreement, 1 for identical and 0 for maximally different."""
    a, b = bipolarize(a), bipolarize(b)
    if a.size != b.size:
        raise ValueError("patterns must have equal size")
    return float(np.dot(a, b) / (2.0 * a.size) + 0.5)

def potentials(input_pattern, prototypes):
    """Hamming-layer potentials: normalized agreement with each prototype."""
    return np.array([similarity(input_pattern, p) for p in prototypes], dtype=float)

