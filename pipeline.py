import argparse
import json
import os

import dataset
from runClustering import runClustering, saveResults
from evaluation import loadLabels, evaluate, precisionAtK


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Run the full insider detection pipeline end to end."
    )
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", default="outputs/syntheticOrgLog.csv")
    parser.add_argument("--labels", default="outputs/groundTruthLabels.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def runEvaluation(labelsPath, allID, labels, clusterTable, nodeScores):
    yTrue = loadLabels(allID, labelsPath)
    topCluster = int(clusterTable.iloc[0]["clusterId"])
    flagged = (labels == topCluster).astype(int)
    metrics = evaluate(yTrue, flagged, nodeScores)
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics["topClusterId"] = topCluster
    metrics["nInsiders"] = int(yTrue.sum())
    metrics["precisionAtNInsiders"] = float(
        precisionAtK(yTrue, nodeScores, max(int(yTrue.sum()), 1))
    )
    return metrics


def main():
    args = parseArgs()

    if args.generate:
        dataset.main()

    allID, labels, clusterTable, nodeScores = runClustering(args.log, args.k, args.seed)
    clusterPath, nodePath = saveResults(args.output_dir, allID, labels, clusterTable, nodeScores)

    print(f"Nodes: {len(allID)}  Clusters: {len(clusterTable)}  k: {args.k}")
    print(clusterTable[["clusterId", "size", "anomalyScore"]].to_string(index=False))
    print(f"Saved {clusterPath}")
    print(f"Saved {nodePath}")

    if not args.skip_eval:
        metrics = runEvaluation(args.labels, allID, labels, clusterTable, nodeScores)
        metricsPath = os.path.join(args.output_dir, "metrics.json")
        with open(metricsPath, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Evaluation:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print(f"Saved {metricsPath}")


if __name__ == "__main__":
    main()
