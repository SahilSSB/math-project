import pandas as pd
import numpy as np


def adjacency():
    df = pd.read_csv("./outputs/synthetic_org_log.csv", usecols=["sender_id", "recipient_id"])

    allID = sorted(pd.concat([df["sender_id"], df["recipient_id"]]).unique())
    n = len(allID)
    idToIdx = {empID: idx for idx, empID in enumerate(allID)}
    adjacencyMatrix = np.zeros((n, n))
    for row in df.itertuples(index=False):
        i = idToIdx[row.sender_id]
        j = idToIdx[row.recipient_id]
        adjacencyMatrix[i][j] += 1
        adjacencyMatrix[j][i] += 1

    return adjacencyMatrix, allID, idToIdx 
