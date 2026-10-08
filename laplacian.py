import numpy as np
import adjacency as a
import degree as d


def laplacian(adjacencyMatrix, degrees, degreeMatrix):
    laplacianMatrix = degreeMatrix - adjacencyMatrix

    n = adjacencyMatrix.shape[0]
    inverse = np.zeros(n, dtype=float)
    np.divide(1.0, np.sqrt(degrees), out=inverse, where=degrees > 0)
    invSqrtDeg = np.diag(inverse)
    identity = np.diag((degrees > 0).astype(float))
    symmLap = identity - invSqrtDeg @ adjacencyMatrix @ invSqrtDeg

    return laplacianMatrix, symmLap 
