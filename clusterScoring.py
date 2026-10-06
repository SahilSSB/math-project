"""Explainable graph-only cluster scores; no ground-truth labels are read."""

import numpy as np
import pandas as pd


def clusterScoring(adjacencyMatrix, clusterLabels):
    """Return (ranked cluster table, scores aligned with the input nodes).

    score = density * internalShare * frequencyFactor
    frequencyFactor = mean internal edge weight / (that mean + global mean)
    Internal weight counts each undirected edge once; internalShare uses
    twice that weight divided by the sum of member degrees.
    Scores are relative heuristics, not probabilities of insider activity.
    """
    A = np.asarray(adjacencyMatrix, dtype=float)
    labels = np.asarray(clusterLabels)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not A.size:
        raise ValueError("A must be a non-empty square matrix.")
    if not np.isfinite(A).all() or (A < 0).any() or not np.allclose(A, A.T):
        raise ValueError("A must be finite, non-negative and symmetric.")
    if not np.allclose(np.diag(A), 0):
        raise ValueError("A must have a zero diagonal (no self-edges).")
    if labels.shape != (len(A),) or labels.dtype.kind not in "iu":
        raise ValueError("Provide one integer cluster label per node.")

    edges = A[np.triu_indices(len(A), k=1)]
    positiveEdges = edges[edges > 0]
    globalMean = float(positiveEdges.mean()) if positiveEdges.size else 0.0
    nodeScores = np.zeros(len(A))
    rows = []
    for clusterId in np.unique(labels):
        members = np.flatnonzero(labels == clusterId)
        size = len(members)
        subgraph = A[np.ix_(members, members)]
        internalEdges = subgraph[np.triu_indices(size, k=1)]
        edgeCount = int(np.count_nonzero(internalEdges))
        internalWeight = float(internalEdges.sum())
        externalWeight = float(A[np.ix_(members, np.flatnonzero(labels != clusterId))].sum())
        possibleEdges = size * (size - 1) / 2
        density = edgeCount / possibleEdges if possibleEdges else 0.0
        volume = 2 * internalWeight + externalWeight
        internalShare = 2 * internalWeight / volume if volume else 0.0
        meanWeight = internalWeight / edgeCount if edgeCount else 0.0
        frequencyFactor = meanWeight / (meanWeight + globalMean) if meanWeight else 0.0
        score = density * internalShare * frequencyFactor
        nodeScores[members] = score
        rows.append({
            "clusterId": int(clusterId), "size": size,
            "internalEdges": edgeCount, "internalWeight": internalWeight,
            "externalWeight": externalWeight, "density": density,
            "internalShare": internalShare, "meanInternalWeight": meanWeight,
            "frequencyFactor": frequencyFactor, "anomalyScore": score,
        })
    table = pd.DataFrame(rows).sort_values(
        ["anomalyScore", "clusterId"], ascending=[False, True]
    ).reset_index(drop=True)
    return table, nodeScores
