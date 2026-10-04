import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def loadLabels(allID, path="outputs/groundTruthLabels.csv"):
    gt = pd.read_csv(path).set_index("nodeId")
    return gt.loc[allID, "isInsider"].astype(int).to_numpy()


def evaluate(yTrue, flagged, scores):
    return {
        "flagged": int(flagged.sum()),
        "precision": precision_score(yTrue, flagged, zero_division=0),
        "recall": recall_score(yTrue, flagged, zero_division=0),
        "f1": f1_score(yTrue, flagged, zero_division=0),
        "rocAuc": roc_auc_score(yTrue, scores),
        "avgPrecision": average_precision_score(yTrue, scores),
    }


def precisionAtK(yTrue, scores, k):
    top = np.argsort(-scores)[:k]
    return yTrue[top].sum() / k
