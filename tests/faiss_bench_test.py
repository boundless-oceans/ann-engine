"""可选 Faiss 基线测试，在隔离虚拟环境中手动执行，不作为 C++ 构建依赖。"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench/faiss_bench.py"
SPEC = importlib.util.spec_from_file_location("faiss_bench", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class FaissBenchTest(unittest.TestCase):
    def test_vector_readers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.vecs"
            values = np.array([[1, 2], [-3, 4]], dtype=np.float32)
            rows = np.zeros((2, 3), dtype="<i4")
            rows[:, 0] = 2
            rows[:, 1:] = values.view("<i4")
            rows.tofile(path)
            actual = BENCH.read_vectors(path, True)
            np.testing.assert_array_equal(actual, values)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertTrue(actual.flags.owndata)
            rows[:, 1:] = [[1, 2], [3, 4]]
            rows.tofile(path)
            np.testing.assert_array_equal(BENCH.read_vectors(path, False), [[1, 2], [3, 4]])

    def test_invalid_vector_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.vecs"
            for raw in (b"", b"abc", np.array([0], dtype="<i4").tobytes(),
                        np.array([2, 1], dtype="<i4").tobytes(),
                        np.array([1, 7, 2, 8], dtype="<i4").tobytes()):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    BENCH.read_vectors(path, True)

    def test_strict_recall_excludes_missing_and_keeps_id_policy(self):
        got = np.array([[0, 2, -1], [3, 5, 4]])
        reference = np.array([[0, 1, 2], [3, 4, 6]])
        self.assertAlmostEqual(BENCH.strict_recall(got, reference), 4 / 6)
        self.assertEqual(BENCH.strict_recall(np.array([[9]]), np.array([[8]])), 0)

    def test_recall_rejects_duplicates_and_wrong_shapes(self):
        for got, reference in ((np.array([[0, 0]]), np.array([[0, 1]])),
                               (np.zeros((1, 2)), np.zeros((1, 1))),
                               (np.zeros((0, 2)), np.zeros((0, 2)))):
            with self.assertRaises(ValueError):
                BENCH.strict_recall(got, reference)

    def test_nearest_rank_statistics(self):
        stats = BENCH.latency_stats(np.array([4, 1, 3, 2]))
        self.assertEqual(stats["mean_ms"], 2.5)
        self.assertEqual(stats["p50_ms"], 2)
        self.assertEqual(stats["p95_ms"], 4)
        self.assertEqual(stats["p99_ms"], 4)
        with self.assertRaises(ValueError):
            BENCH.latency_stats(np.array([]))

    def test_ef_parser(self):
        self.assertEqual(BENCH.ef_list("4,16,4"), [4, 16, 4])
        for value in ("", "0", "-1", "1,", "1,,2", "2x", "2147483648", "1, 2"):
            with self.assertRaises(argparse.ArgumentTypeError):
                BENCH.ef_list(value)

    def test_seeded_build_and_single_query_measurement(self):
        vectors = np.array([[0, 0], [1, 0], [0, 2], [-3, 1]], dtype=np.float32)
        index, _ = BENCH.build_index(vectors, 2, 16, 42)
        repeated, _ = BENCH.build_index(vectors, 2, 16, 42)
        before = faiss.vector_to_array(index.hnsw.neighbors).copy()
        index.hnsw.efSearch = 16
        metrics, ids = BENCH.measure(index, vectors, 4, 1, 2)
        self.assertGreater(metrics["qps"], 0)
        self.assertEqual(ids.shape, (4, 4))
        np.testing.assert_array_equal(ids[:, 0], np.arange(4))
        np.testing.assert_array_equal(before, faiss.vector_to_array(index.hnsw.neighbors))
        np.testing.assert_array_equal(before, faiss.vector_to_array(repeated.hnsw.neighbors))
        self.assertGreater(BENCH.logical_payload_bytes(index), vectors.nbytes)
        self.assertEqual(faiss.omp_get_max_threads(), 1)

    def test_cli_repeated_ef_csv_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "rows.csv"
            metadata = Path(folder) / "context.json"
            result = subprocess.run(
                [sys.executable, "-I", str(SCRIPT), "--synthetic", "--n", "64", "--nq", "5",
                 "--dim", "7", "--k", "3", "--m", "4", "--ef-search-list", "4,16,4",
                 "--warmup", "0", "--runs", "1", "--csv", str(output),
                 "--metadata", str(metadata)], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr.count("Build completed"), 1)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["recall"], rows[2]["recall"])
            self.assertEqual(rows[0]["memory_metric"], "logical_payload_bytes")
            context = json.loads(metadata.read_text())
            self.assertTrue(context["isolated"])
            self.assertEqual(context["omp_threads"], 1)
            self.assertEqual(context["query_batch_size"], 1)
            self.assertEqual(context["groundtruth"], "independent Flat")


if __name__ == "__main__":
    unittest.main()
