import argparse
import os

import pandas as pd

from adjacency import adjacency
from degree import degree
from laplacian import laplacian
from eigenvalueDecomposer import eigenvalueDecomposer
from UMatrixBuilder import UMatrix
from clustering import clustering
from clusterScoring import clusterScoring


def runClustering(logPath, k, seed):
    A, allID, idToIdx = adjacency(logPath)
    degrees, degreeMatrix = degree(A)
    laplacianMatrix, symmLap = laplacian(A, degrees, degreeMatrix)
    eigenValues, eigenVectors = eigenvalueDecomposer(symmLap)
    U = UMatrix(eigenVectors, k)
    labels = clustering(U, k, randomState=seed)
    clusterTable, nodeScores = clusterScoring(A, labels)
    return allID, labels, clusterTable, nodeScores


def saveResults(outputDir, allID, labels, clusterTable, nodeScores):
    os.makedirs(outputDir, exist_ok=True)
    clusterPath = os.path.join(outputDir, "clusterScores.csv")
    nodePath = os.path.join(outputDir, "nodeClusters.csv")
    clusterTable.to_csv(clusterPath, index=False)
    pd.DataFrame({
        "nodeId": allID,
        "clusterId": labels,
        "anomalyScore": nodeScores,
    }).to_csv(nodePath, index=False)
    return clusterPath, nodePath


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Run spectral clustering and cluster scoring on a communication log."
    )
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--log", default="outputs/syntheticOrgLog.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parseArgs()
    allID, labels, clusterTable, nodeScores = runClustering(args.log, args.k, args.seed)
    clusterPath, nodePath = saveResults(args.output_dir, allID, labels, clusterTable, nodeScores)

    print(f"Nodes: {len(allID)}  Clusters: {len(clusterTable)}  k: {args.k}")
    print(clusterTable[["clusterId", "size", "anomalyScore"]].to_string(index=False))
    print(f"Saved {clusterPath}")
    print(f"Saved {nodePath}")


if __name__ == "__main__":
    main()
