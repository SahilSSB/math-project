# Insider Thread Detection with Spectral Clustering

Detects a hidden group of colluding insiders in an organization's communication
log by treating the log as a graph and clustering it with spectral methods.

## Pipeline Overview

```
dataset.py → adjacency.py → degree.py → laplacian.py → eigenvalueDecomposer.py → UMatrixBuilder.py → (clustering, scoring, evaluation)
```

## 1. Data Generation (`dataset.py`)

Builds a synthetic organizational communication log (email / file-transfer
metadata) with a known, planted insider ring.

**How the graph is built**

- **Background structure:** Barabási–Albert graph, giving a few high-degree hubs
  (like managers) and mostly low-degree employees.
- **Noise:** a sparse Erdős–Rényi graph is merged in to simulate random one-off
  messages.
- **Planted insider ring:** a small, near-complete subgraph where each possible
  pair is connected with probability `CLIQUE_DENSITY`.
- **Event log:** every edge becomes timestamped transfer events. Insider edges
  have more events (higher Poisson rate) but much smaller byte sizes than normal
  edges.

**Default configuration**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `N_EMPLOYEES` | 500 | Organization size |
| `BA_M` | 3 | BA attachment parameter |
| `ER_NOISE_P` | 0.002 | Random noise edge probability |
| `N_CLIQUES` | 1 | Number of planted insider rings |
| `CLIQUE_SIZE` | 6 | Insiders per ring |
| `CLIQUE_DENSITY` | 0.85 | Edge probability inside the ring |
| `SIM_DAYS` | 60 | Simulated time span |
| `RANDOM_SEED` | 42 | Reproducibility |

**Outputs** (written to `outputs/`)

- `syntheticOrgLog.csv`: `senderId, recipientId, timestamp, bytes`. This is
  the only file the detector reads.
- `groundTruthLabels.csv`: `nodeId, isInsider, cliqueId`. Kept separate and
  used only for evaluation.
- `datasetMetadata.json`: generation parameters and the insider node IDs.

---

## 2. Adjacency Matrix (`adjacency.py`)

`adjacency()` reads the log and builds a symmetric, weighted adjacency matrix `A`.
Each message between employees *i* and *j* adds 1 to both `A[i][j]` and `A[j][i]`,
so an entry is the total number of messages exchanged between two people,
regardless of direction.

Returns `(adjacencyMatrix, allID, idToIdx)`. `allID` is the sorted list of
employee IDs and `idToIdx` maps an ID to its row/column index.

---

## 3. Degree Matrix (`degree.py`)

`degree(A)` returns the weighted degree of each node (row sums of `A`) and the
diagonal degree matrix `D`.

---

## 4. Graph Laplacians (`laplacian.py`)

`laplacian(A, degrees, D)` returns two matrices:

- **Unnormalized Laplacian:** `L = D − A`
- **Symmetric normalized Laplacian:** `LSym = I − D^(−1/2) A D^(−1/2)`

The pipeline uses `LSym` for the eigendecomposition. Dividing by `sqrt(degree)`
assumes every node has degree > 0, which holds here because every employee
appears in the log.

---

## 5. Eigendecomposition (`eigenvalueDecomposer.py`)

`eigenvalueDecomposer(symmLap)` calls `numpy.linalg.eigh`, which is suited to
symmetric matrices. It returns eigenvalues in ascending order and the matching
eigenvectors as columns. The eigenvectors for the smallest eigenvalues carry the
cluster structure of the graph: nodes that are densely connected get similar
values in these vectors.

---

## 6. Spectral Embedding (`UMatrixBuilder.py`)

`UMatrix(eigenVectors, k)` takes the first `k` eigenvectors as an n×k matrix and
normalizes each row to unit length (Ng Jordan Weiss style). Each employee is now a
point in k dimensional space, ready for clustering. Zero norm rows are guarded with
a small epsilon to avoid division by zero.

---

## Setup and Usage

**Requirements:** Python 3.9+ with `numpy`, `pandas`, `networkx`.

```bash
pip install numpy pandas networkx

# 1. Generate the dataset (creates the outputs/ folder)
python dataset.py

# 2. Inspect the Laplacian matrices
python laplacian.py
```

---

## TODO 

- [ ] Clustering (k-means on `U`)
- [ ] Cluster scoring (anomaly score)
- [ ] Evaluation (precision / recall against ground truth)
- [ ] Main pipeline and how to run it end to end
- [ ] Results, discussion, limitations
