"""Small runnable example for the clustering and cluster-scoring tasks."""

import argparse
from pathlib import Path

import pandas as pd

from adjacency import adjacency
from degree import degree
from laplacian import laplacian
from eigenvalueDecomposer import eigenvalueDecomposer
from UMatrixBuilder import UMatrix
from clustering import clustering
from clusterScoring import clusterScoring


def run(logPath="outputs/syntheticOrgLog.csv", k=2, randomState=42):
    A, allID, _ = adjacency(logPath)
    if not 2 <= k < len(A):
        raise ValueError("Choose 2 <= k < number of observed employees.")
    degrees, D = degree(A)
    _, LSym = laplacian(A, degrees, D)
    _, eigenVectors = eigenvalueDecomposer(LSym)
    labels = clustering(UMatrix(eigenVectors, k), k, randomState)
    clusters, scores = clusterScoring(A, labels)
    nodes = pd.DataFrame({"nodeId": allID, "clusterId": labels, "anomalyScore": scores})
    return clusters, nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="outputs/syntheticOrgLog.csv")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    clusters, nodes = run(args.log, args.k, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(args.output_dir / "clusterScores.csv", index=False)
    nodes.to_csv(args.output_dir / "nodeClusters.csv", index=False)
    print(clusters.to_string(index=False))
    print(f"\nSaved clusterScores.csv and nodeClusters.csv in {args.output_dir}")


if __name__ == "__main__":
    main()
