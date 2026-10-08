# Insider Thread Detection with Spectral Clustering

Detects a hidden group of colluding insiders in an organization's communication
log by treating the log as a graph and clustering it with spectral methods.

## Pipeline Overview

```
dataset.py → adjacency.py → degree.py → laplacian.py → eigenvalueDecomposer.py → UMatrixBuilder.py → clustering.py → clusterScoring.py
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
regardless of direction. Self-messages are ignored. An optional `path` argument
selects another log with the same column names.

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

The pipeline uses `LSym` for the eigendecomposition. Zero-degree nodes have zero
rows and columns in `LSym`, avoiding division by zero. Only employees observed in
the event log enter the detector; generated employees with no events are absent.

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

## 7. Clustering (`clustering.py`)

`clustering(U, k, randomState=42)` returns an integer cluster label for each row
of `U`. Labels retain the original employee order; their numeric values do not
indicate suspicion. It uses [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
with 20 initializations and a fixed random seed. Invalid inputs and fewer than
`k` distinct embedding rows are rejected.

The example uses `k=2` as an initial partition, not as a claim that every graph
has two natural groups. Set `--k` to explore other partitions; this also changes
the number of eigenvectors used. Do not select `k` using evaluation labels and
then present performance on those same labels as independent validation.

## 8. Cluster Scoring (`clusterScoring.py`)

`clusterScoring(A, clusterLabels)` returns a DataFrame sorted by decreasing
`anomalyScore`, plus a score array aligned with the rows of `A`. Each node receives
its cluster's score. The scorer reads neither ground truth nor dataset metadata.

For each cluster C:

- `density` = observed internal edges / possible internal edges.
- `internalShare` = 2 × internal message weight / sum of member degrees.
- `meanInternalWeight` = internal message weight / observed internal edges.
- `frequencyFactor` = meanInternalWeight / (meanInternalWeight + global mean edge weight).
- **`anomalyScore = density × internalShare × frequencyFactor`.**

Internal edges are counted once. The global mean is over observed undirected
edges, not all possible pairs. All components and the final score are included
in the cluster table. Singletons and clusters without internal edges score zero.
Ties are ordered by cluster ID. The score lies between 0 and 1 and favors dense,
frequently communicating groups with relatively little external communication.

This is a documented heuristic, not a probability or a validated accusation.
Legitimate close teams can rank highly, and clustering can mix insiders with
ordinary employees. It does not use byte sizes or timestamps. It ranks all
clusters without automatically declaring the top cluster malicious; threshold
selection and the full evaluation/report remain separate work.

## Setup and Usage

**Requirements:** Python 3.9+ and the packages in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 1. Generate the dataset (creates the outputs/ folder)
python dataset.py

# 2. Run the clustering/scoring example from the repository directory
python runClustering.py --k 2

```

`runClustering.py` connects the existing spectral stages to these two tasks.
It saves `outputs/clusterScores.csv` (ranked clusters with score components) and
`outputs/nodeClusters.csv` (`nodeId, clusterId, anomalyScore`). Optional flags:
`--log PATH`, `--seed INTEGER`, and `--output-dir PATH`. It does not load labels
or run evaluation. Re-running the command replaces these two result files.

## TODO 

- [x] Clustering (k-means on `U`)
- [x] Cluster scoring (anomaly score)
- [ ] Evaluation (precision / recall against ground truth)
- [ ] Main pipeline and how to run it end to end
- [ ] Results, discussion, limitations
