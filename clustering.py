"""K-means on the row-normalized spectral embedding."""

import numpy as np
from sklearn.cluster import KMeans


def clustering(U, k, randomState=42):
    """Return one integer cluster label per row of U, in the same order."""
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or not U.size or not np.isfinite(U).all():
        raise ValueError("U must be a non-empty, finite 2D matrix.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer.")
    if not 1 <= k <= len(U):
        raise ValueError("k must be between 1 and the number of nodes.")
    if len(np.unique(U, axis=0)) < k:
        raise ValueError("U must contain at least k distinct rows.")
    return KMeans(n_clusters=k, n_init=20, random_state=randomState).fit_predict(U)
