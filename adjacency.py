import pandas as pd
import numpy as np


def adjacency(path="outputs/syntheticOrgLog.csv"):
    df = pd.read_csv(path, usecols=["senderId", "recipientId"])
    if df.empty or df.isna().any().any():
        raise ValueError("The log must contain non-missing sender and recipient IDs.")

    allID = sorted(pd.concat([df["senderId"], df["recipientId"]]).unique())
    n = len(allID)
    idToIdx = {empID: idx for idx, empID in enumerate(allID)}
    adjacencyMatrix = np.zeros((n, n))
    for row in df.itertuples(index=False):
        i = idToIdx[row.senderId]
        j = idToIdx[row.recipientId]
        if i == j:  # Self-messages do not connect different employees.
            continue
        adjacencyMatrix[i][j] += 1
        adjacencyMatrix[j][i] += 1

    return adjacencyMatrix, allID, idToIdx 
