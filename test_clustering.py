import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adjacency import adjacency
from clustering import clustering
from clusterScoring import clusterScoring
from degree import degree
from evaluation import loadLabels, evaluate, precisionAtK
from laplacian import laplacian
from runClustering import run


class ClusteringTests(unittest.TestCase):
    def test_separated_groups_and_reproducibility(self):
        U = np.array([[1, 0], [.99, .01], [0, 1], [.01, .99]])
        labels = clustering(U, 2)
        np.testing.assert_array_equal(labels, clustering(U, 2))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_invalid_embedding(self):
        for U, k in [([], 2), ([[np.nan]], 1), ([[1], [1]], 2), ([[1]], 2), ([[1]], 1.5)]:
            with self.assertRaises(ValueError):
                clustering(U, k)

    def test_hand_calculated_scores_and_node_order(self):
        A = np.array([[0, 10, 1, 0], [10, 0, 0, 0],
                      [1, 0, 0, 2], [0, 0, 2, 0]])
        table, scores = clusterScoring(A, np.array([7, 7, 3, 3]))
        # Global mean = 13/3; density = 1 for both two-node clusters.
        expected = (20 / 21) * (10 / (10 + 13 / 3))
        self.assertAlmostEqual(scores[0], expected)
        self.assertEqual(table.iloc[0].clusterId, 7)
        np.testing.assert_allclose(scores, [expected, expected, 4/5 * 6/19, 4/5 * 6/19])

    def test_singletons_and_edgeless_graph(self):
        table, scores = clusterScoring(np.zeros((3, 3)), np.array([0, 1, 1]))
        np.testing.assert_array_equal(scores, np.zeros(3))
        self.assertTrue(np.isfinite(table.to_numpy()).all())

    def test_invalid_graphs(self):
        for A in [np.eye(2), [[0, 1], [0, 0]], [[0, -1], [-1, 0]], [[0, np.nan], [np.nan, 0]]]:
            with self.assertRaises(ValueError):
                clusterScoring(A, [0, 1])
        with self.assertRaises(ValueError):
            clusterScoring(np.zeros((2, 2)), [0])

    def test_isolated_node_laplacian(self):
        A = np.array([[0., 1, 0], [1, 0, 0], [0, 0, 0]])
        degrees, D = degree(A)
        _, normalized = laplacian(A, degrees, D)
        np.testing.assert_allclose(normalized, [[1, -1, 0], [-1, 1, 0], [0, 0, 0]])

    def test_other_fork_evaluation_compatibility(self):
        A = np.array([[0, 10, 1, 0], [10, 0, 0, 0],
                      [1, 0, 0, 2], [0, 0, 2, 0]])
        labels = np.array([7, 7, 3, 3])
        clusters, scores = clusterScoring(A, labels)
        # The truth file is deliberately shuffled; align by ID, not CSV row.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groundTruthLabels.csv"
            pd.DataFrame({"nodeId": ["c", "a", "d", "b"],
                          "isInsider": [0, 1, 0, 0]}).to_csv(path, index=False)
            truth = loadLabels(["a", "b", "c", "d"], path)
        np.testing.assert_array_equal(truth, [1, 0, 0, 0])
        flagged = labels == int(clusters.iloc[0].clusterId)
        metrics = evaluate(truth, flagged, scores)
        self.assertEqual(metrics["flagged"], 2)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(precisionAtK(truth, scores, 2), 0.5)

    def test_log_to_clusters_without_ground_truth(self):
        events = []
        # Two disconnected triangles, with different message frequencies.
        for group, count in [(["a", "b", "c"], 10), (["d", "e", "f"], 1)]:
            for i in range(3):
                for j in range(i + 1, 3):
                    events.extend([{"senderId": group[i], "recipientId": group[j]}] * count)
        events.append({"senderId": "a", "recipientId": "a"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.csv"
            pd.DataFrame(events).to_csv(path, index=False)
            A, ids, _ = adjacency(path)
            self.assertEqual(ids, list("abcdef"))
            self.assertEqual(A[0, 1], 10)
            self.assertEqual(A[0, 0], 0)
            clusters, nodes = run(path)
        topId = clusters.iloc[0].clusterId
        self.assertEqual(set(nodes.loc[nodes.clusterId == topId, "nodeId"]), set("abc"))
        self.assertTrue(nodes.anomalyScore.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
