"""从 ann_bench CSV 导出曲线；仅绘图需要 matplotlib，C++ 构建与测试不依赖它。"""

import argparse
import csv
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot a single-graph HNSW efSearch sweep")
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context", default="See bench/hnsw.md for measurement environment")
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        parser.error("CSV contains no measurements")
    # 防止把多个规模或多次建图的行误连成同一条曲线。
    identity = ("index", "metric", "n", "dim", "nq", "k", "M", "Mmax0",
                "ef_construction", "seed", "warmup", "runs", "build_seconds", "memory_bytes")
    first = rows[0]
    if first["index"] != "HNSW" or any(
        row[field] != first[field] for row in rows for field in identity
    ):
        parser.error("CSV must describe one HNSW graph and one query workload")
    ef = [int(row["effective_ef_search"]) for row in rows]
    recall = [float(row["recall"]) for row in rows]
    qps = [float(row["qps"]) for row in rows]
    p99 = [float(row["p99_ms"]) for row in rows]
    if any(value <= 0 for value in ef) or any(
        not math.isfinite(value) for values in (recall, qps, p99) for value in values
    ) or any(value < 0 or value > 1 for value in recall) or any(value <= 0 for value in qps):
        parser.error("CSV contains invalid measurement values")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "ann-engine-hnsw-sweep"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(recall, qps, "o-", color="#176b8f")
    for x, y, width in zip(recall, qps, ef):
        axes[0].annotate(f"ef={width}", (x, y), xytext=(6, 5), textcoords="offset points")
    axes[0].set(xlabel=f"Strict Recall@{first['k']}", ylabel="Queries / second")
    axes[0].margins(x=0.16, y=0.18)
    axes[1].plot(ef, p99, "o-", color="#a5522a")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(sorted(set(ef)), labels=[str(v) for v in sorted(set(ef))])
    axes[1].set(xlabel="Effective efSearch", ylabel="P99 latency (ms)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"HNSW: n={first['n']}, dim={first['dim']}, queries={first['nq']}, "
                 f"M={first['M']}, efConstruction={first['ef_construction']}")
    fig.text(0.5, 0.02, args.context, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(args.output, dpi=160, metadata={"Date": None} if args.output.suffix == ".svg" else {})
    plt.close(fig)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
