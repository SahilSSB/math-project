import numpy as np
import adjacency as a
import degree as d
import laplacian as lap
import eigenvalueDecomposer as eig


def UMatrix(eigenVectors, k, skipFirst=True, normalize=False):
    start = 1 if skipFirst else 0
    U = eigenVectors[:, start:k].copy()
    if normalize:
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        U = U / norms
    return U

