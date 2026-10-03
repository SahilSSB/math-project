import numpy as np
import adjacency as a
import degree as d
import laplacian as lap


def eigenvalueDecomposer(symmLap):
    eigenValues, eigenVectors = np.linalg.eigh(symmLap)
    return eigenValues, eigenVectors

