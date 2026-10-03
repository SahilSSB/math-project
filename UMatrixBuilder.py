import numpy as np
import adjacency as a
import degree as d
import laplacian as lap
import eigenvalueDecomposer as eig


def UMatrix(eigenVectors, k):
    U = eigenVectors[:, :k].copy()   
    rowNormalization = np.linalg.norm(U, axis=1, keepdims=True) 

    rowNormalization[rowNormalization == 0] = 1e-10

    UNormalized = U / rowNormalization
    return UNormalized

