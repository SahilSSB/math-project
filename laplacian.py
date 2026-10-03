import numpy as np
import adjacency as a
import degree as d


def laplacian(adjacencyMatrix, degrees, degreeMatrix):
    laplacianMatrix = degreeMatrix - adjacencyMatrix

    n = adjacencyMatrix.shape[0]
    invSqrtDeg = np.diag(1.0 / np.sqrt(degrees))
    identity = np.eye(n)
    symmLap = identity - invSqrtDeg @ adjacencyMatrix @ invSqrtDeg

    return laplacianMatrix, symmLap 
