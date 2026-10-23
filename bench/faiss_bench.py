"""Python Faiss CPU 基线：单查询调用、同图 ef 扫描、严格 ID Recall。"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

# 仅设置本进程的线程预算，必须先于 NumPy/Faiss 导入。
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import faiss
import numpy as np


FIELDS = (
    "index", "metric", "n", "dim", "nq", "k", "M", "Mmax0", "ef_construction",
    "ef_search", "effective_ef_search", "seed", "warmup", "runs", "build_seconds",
    "memory_bytes", "recall", "qps", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
    "memory_metric", "faiss_version", "numpy_version", "threads", "query_batch_size",
)


def positive(value):
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 2147483647:
        raise argparse.ArgumentTypeError("expected a positive integer <= INT32_MAX")
    return int(value)


def ef_list(value):
    return [positive(part) for part in value.split(",")]


def read_vectors(path, floating):
    """检查逐行维度后拷贝有效数据，返回独立连续数组，不长期持有 mmap。"""
    if path.stat().st_size == 0 or path.stat().st_size % 4:
        raise ValueError(f"invalid vector file length: {path}")
    raw = np.memmap(path, mode="r", dtype="<i4")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1):
        raise ValueError(f"invalid vector row size: {path}")
    rows = raw.reshape(-1, dim + 1)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"inconsistent vector dimensions: {path}")
    data = rows[:, 1:].view("<f4") if floating else rows[:, 1:]
    return np.array(data, dtype=np.float32 if floating else np.int64, order="C", copy=True)


def strict_recall(ids, reference):
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0 or reference.ndim != 2:
        raise ValueError("recall requires nonempty 2D arrays")
    nq, k = ids.shape
    if reference.shape[0] != nq or reference.shape[1] < k:
        raise ValueError("reference shape does not cover queries and k")
    ordered = np.sort(ids, axis=1)
    if np.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)):
        raise ValueError("duplicate result IDs")
    matches = (ids[:, :, None] == reference[:, None, :k]) & (ids[:, :, None] >= 0)
    return float(matches.any(axis=2).sum()) / (nq * k)


def latency_stats(values):
    ordered = np.sort(values)
    if ordered.size == 0:
        raise ValueError("latency samples cannot be empty")

    def percentile(p):
        return float(ordered[math.ceil(p * ordered.size) - 1])

    return dict(mean_ms=float(ordered.mean()), p50_ms=percentile(0.50),
                p95_ms=percentile(0.95), p99_ms=percentile(0.99), max_ms=float(ordered[-1]))


def build_index(base, m, ef_construction, seed):
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(base.shape[1], m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.rng = faiss.RandomGenerator(seed)
    # 默认策略显式记录，不为追平自研图而修改 Faiss 的内部算法。
    index.hnsw.search_bounded_queue = True
    index.hnsw.check_relative_distance = True
    index.keep_max_size_level0 = False
    started = time.perf_counter_ns()
    index.add(base)
    seconds = (time.perf_counter_ns() - started) / 1e9
    return index, seconds


def logical_payload_bytes(index):
    # 这里只统计可见数组的有效元素，不冒充 C++ 分配容量或整个进程 RSS。
    graph = index.hnsw
    return (int(index.ntotal) * index.d * 4 + graph.levels.size() * 4
            + graph.offsets.size() * 8 + graph.neighbors.size() * 4
            + graph.cum_nneighbor_per_level.size() * 4 + graph.assign_probas.size() * 8)


def measure(index, queries, k, warmup, runs):
    # 查询切片与输出缓冲在计时外准备，避免每次分配 Python 结果数组。
    prepared = [queries[q:q + 1] for q in range(len(queries))]
    distances = np.empty((1, k), dtype=np.float32)
    ids = np.empty((1, k), dtype=np.int64)
    samples = np.empty(len(queries) * runs, dtype=np.float64)

    for _ in range(warmup):
        for query in prepared:
            index.search(query, k, D=distances, I=ids)
    elapsed = 0
    position = 0
    for _ in range(runs):
        started = time.perf_counter_ns()
        for query in prepared:
            before = time.perf_counter_ns()
            index.search(query, k, D=distances, I=ids)
            after = time.perf_counter_ns()
            samples[position] = (after - before) / 1e6
            position += 1
        elapsed += time.perf_counter_ns() - started

    # 额外一轮只收集 ID 统计召回，不计入正式吞吐或延迟。
    results = np.empty((len(queries), k), dtype=np.int64)
    for q, query in enumerate(prepared):
        index.search(query, k, D=distances, I=ids)
        results[q] = ids[0]
    return dict(qps=len(samples) / (elapsed / 1e9), **latency_stats(samples)), results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-query Faiss HNSW CPU baseline")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, default=Path("data"))
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n", type=positive, default=1000)
    parser.add_argument("--nq", type=positive, default=100)
    parser.add_argument("--dim", type=positive, default=16)
    parser.add_argument("--max-base", type=positive)
    parser.add_argument("--k", type=positive, default=10)
    parser.add_argument("--m", type=positive, default=16)
    parser.add_argument("--ef-construction", type=positive, default=200)
    parser.add_argument("--ef-search-list", type=ef_list, default=ef_list("16,32,64,128,256"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=positive, default=3)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.m < 2 or args.warmup < 0 or not 0 <= args.seed <= 2147483647:
        parser.error("require M >= 2, warmup >= 0, and 0 <= seed <= INT32_MAX")
    faiss.omp_set_num_threads(1)
    print("[faiss] Loading dataset", file=sys.stderr, flush=True)
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        base = rng.uniform(-1, 1, (args.n, args.dim)).astype(np.float32)
        queries = rng.uniform(-1, 1, (args.nq, args.dim)).astype(np.float32)
        reference = None
        reference_kind = "independent Flat"
    else:
        base = read_vectors(args.data / "sift_base.fvecs", True)
        queries = read_vectors(args.data / "sift_query.fvecs", True)
        reference = read_vectors(args.data / "sift_groundtruth.ivecs", False)
        reference_kind = "official strict ID groundtruth"
        if queries.shape[1] != base.shape[1] or len(reference) != len(queries):
            raise ValueError("dataset dimensions or query counts do not match")
        if reference.shape[1] < args.k or np.any(reference < 0) or np.any(reference >= len(base)):
            raise ValueError("official reference does not cover k or contains invalid IDs")
    if args.max_base and args.max_base < len(base):
        base = base[:args.max_base].copy()
        reference = None
        reference_kind = "independent Flat on truncated base"
    if not np.isfinite(base).all() or not np.isfinite(queries).all():
        raise ValueError("vectors must be finite")
    if reference is None:
        print("[faiss] Computing independent exact Flat reference", file=sys.stderr, flush=True)
        exact = faiss.IndexFlatL2(base.shape[1])
        exact.add(base)
        _, reference = exact.search(queries, args.k)
        del exact
    print(f"[faiss] Building HNSW: {len(base)} vectors", file=sys.stderr, flush=True)
    index, seconds = build_index(base, args.m, args.ef_construction, args.seed)
    print(f"[faiss] Build completed in {seconds:.3f} s", file=sys.stderr, flush=True)
    metadata = dict(python=sys.version, executable=sys.executable,
                    isolated=bool(sys.flags.isolated), numpy=np.__version__, faiss=faiss.__version__,
                    compile_options=faiss.get_compile_options().strip(),
                    omp_threads=faiss.omp_get_max_threads(), query_batch_size=1,
                    timing="Python public search wrapper included; preallocated D/I; no correction",
                    search_bounded_queue=bool(index.hnsw.search_bounded_queue),
                    check_relative_distance=bool(index.hnsw.check_relative_distance),
                    keep_max_size_level0=bool(index.keep_max_size_level0),
                    seed=args.seed, seed_note="Faiss RNG differs from ann-engine mt19937_64",
                    memory_metric="logical_payload_bytes; excludes capacity and object overhead",
                    groundtruth=reference_kind)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    with args.csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        output.flush()
        for requested_ef in args.ef_search_list:
            effective_ef = max(args.k, requested_ef)
            index.hnsw.efSearch = effective_ef
            print(f"[faiss] Measuring ef={requested_ef}, warmup={args.warmup}, runs={args.runs}",
                  file=sys.stderr, flush=True)
            metrics, ids = measure(index, queries, args.k, args.warmup, args.runs)
            row = dict(index="FaissHNSWFlat", metric="L2", n=len(base), dim=base.shape[1],
                       nq=len(queries), k=args.k, M=args.m, Mmax0=index.hnsw.nb_neighbors(0),
                       ef_construction=args.ef_construction, ef_search=requested_ef,
                       effective_ef_search=effective_ef, seed=args.seed, warmup=args.warmup,
                       runs=args.runs, build_seconds=seconds, memory_bytes=logical_payload_bytes(index),
                       memory_metric="logical_payload_bytes", recall=strict_recall(ids, reference),
                       faiss_version=faiss.__version__, numpy_version=np.__version__, threads=1,
                       query_batch_size=1, **metrics)
            writer.writerow(row)
            output.flush()
            print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
