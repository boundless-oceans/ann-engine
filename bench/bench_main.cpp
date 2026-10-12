// 端到端评测程序。
//
// 流程：解析参数 → 准备数据 → 构建索引 → 生成或加载 ground truth
//       → 预热 → 计时测量 → 打印指标。
//
// 用法示例：
//   ann_bench                                  合成数据 100000 × 128，k=10
//   ann_bench --n 1000000 --nq 10000           放大规模
//   ann_bench --data data                      使用磁盘上的 SIFT1M
//   ann_bench --metric ip                      切换距离度量
//
// 注意：所有性能数字必须连同构建配置与硬件环境一起记录，
// 否则数字之间没有可比性。构建配置由 CMake 的配置摘要给出。

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ann/flat.h"
#include "ann/index.h"
#include "ann/types.h"
#include "dataset.h"
#include "metrics.h"

namespace {

struct Options {
    size_t n = 100000;        // 合成数据的 base 向量数
    size_t nq = 1000;         // 查询数
    size_t dim = 128;         // 维度
    size_t k = 10;            // top-k
    ann::Metric metric = ann::Metric::L2;
    std::string data_dir;     // 非空则从磁盘加载，忽略合成数据参数
    size_t max_base = 0;      // 截断 base（调试用）
    uint64_t seed = 42;
    size_t warmup = 1;        // 预热轮数
    size_t runs = 3;          // 正式测量轮数
};

void PrintUsage() {
    std::printf(
        "ann_bench - end-to-end benchmark for ann-engine\n"
        "\n"
        "usage: ann_bench [options]\n"
        "\n"
        "options:\n"
        "  --n <int>         number of base vectors for synthetic data (default 100000)\n"
        "  --nq <int>        number of queries (default 1000)\n"
        "  --dim <int>       vector dimension (default 128)\n"
        "  --k <int>         top-k (default 10)\n"
        "  --metric <name>   l2 | ip (default l2)\n"
        "  --data <dir>      load SIFT1M from dir instead of synthetic data\n"
        "  --max-base <int>  truncate base vectors (debug only)\n"
        "  --seed <int>      random seed for synthetic data (default 42)\n"
        "  --warmup <int>    warmup rounds, not measured (default 1)\n"
        "  --runs <int>      measured rounds (default 3)\n"
        "  -h, --help        show this message\n");
}

// 返回 false 表示参数非法，应当终止。
bool ParseArgs(int argc, char** argv, Options* opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            PrintUsage();
            return false;
        } else if (a == "--n") {
            const char* v = need_value("--n");
            if (v == nullptr) return false;
            opt->n = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--nq") {
            const char* v = need_value("--nq");
            if (v == nullptr) return false;
            opt->nq = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--dim") {
            const char* v = need_value("--dim");
            if (v == nullptr) return false;
            opt->dim = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--k") {
            const char* v = need_value("--k");
            if (v == nullptr) return false;
            opt->k = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--metric") {
            const char* v = need_value("--metric");
            if (v == nullptr) return false;
            if (std::strcmp(v, "l2") == 0) {
                opt->metric = ann::Metric::L2;
            } else if (std::strcmp(v, "ip") == 0) {
                opt->metric = ann::Metric::InnerProduct;
            } else {
                std::fprintf(stderr, "error: unknown metric '%s' (expected l2 or ip)\n", v);
                return false;
            }
        } else if (a == "--data") {
            const char* v = need_value("--data");
            if (v == nullptr) return false;
            opt->data_dir = v;
        } else if (a == "--max-base") {
            const char* v = need_value("--max-base");
            if (v == nullptr) return false;
            opt->max_base = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--seed") {
            const char* v = need_value("--seed");
            if (v == nullptr) return false;
            opt->seed = std::strtoull(v, nullptr, 10);
        } else if (a == "--warmup") {
            const char* v = need_value("--warmup");
            if (v == nullptr) return false;
            opt->warmup = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--runs") {
            const char* v = need_value("--runs");
            if (v == nullptr) return false;
            opt->runs = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            PrintUsage();
            return false;
        }
    }

    if (opt->n == 0 || opt->nq == 0 || opt->dim == 0 || opt->k == 0 || opt->runs == 0) {
        std::fprintf(stderr, "error: n / nq / dim / k / runs must all be positive\n");
        return false;
    }
    return true;
}

const char* MetricName(ann::Metric m) {
    switch (m) {
        case ann::Metric::L2:           return "L2";
        case ann::Metric::InnerProduct: return "InnerProduct";
        case ann::Metric::Cosine:       return "Cosine";
    }
    return "Unknown";
}

// 用索引自身在给定数据上生成 ground truth。
//
// 合成数据没有官方答案，只能用精确索引自算。这条路径同时充当评测框架的自检：
// 用同一份结果既当答案又当被测对象，recall 必然为 1.0，
// 因此它只能验证指标链路是否打通，不能验证索引是否准确。
// 真实准确性必须依靠加载官方 groundtruth（--data）。
std::vector<int32_t> BuildGroundTruth(ann::Index* idx, const float* queries,
                                      size_t nq, size_t k) {
    std::vector<ann::dist_t> dists(nq * k);
    std::vector<ann::idx_t> ids(nq * k);

    ann::ResultView out{dists.data(), ids.data(), k};
    idx->SearchBatch(queries, nq, k, out);

    std::vector<int32_t> gt(nq * k);
    for (size_t i = 0; i < nq * k; ++i) {
        gt[i] = static_cast<int32_t>(ids[i]);
    }
    return gt;
}

// 执行一轮测量：逐条查询计时，同时统计整批耗时。
//
// 每条查询单独取时钟会带来数十纳秒开销。在当前毫秒级的查询耗时下可忽略，
// 但若将来单次查询降到微秒级，应改为「另一轮只测整批」的方式。
void RunRound(const ann::Index& idx, const float* queries, size_t nq, size_t k,
              size_t dim, ann::bench::LatencyRecorder* latency,
              ann::bench::QpsTimer* timer) {
    std::vector<ann::dist_t> dists(k);
    std::vector<ann::idx_t> ids(k);
    ann::ResultView out{dists.data(), ids.data(), k};

    timer->Start();
    for (size_t q = 0; q < nq; ++q) {
        const auto t0 = std::chrono::steady_clock::now();
        idx.Search(queries + q * dim, k, out);
        const auto t1 = std::chrono::steady_clock::now();

        latency->Record(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    timer->Stop();
    timer->AddQueries(nq);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!ParseArgs(argc, argv, &opt)) {
        return 1;
    }

    // ---------------- 准备数据 ----------------
    ann::bench::Dataset ds;
    bool synthetic = opt.data_dir.empty();

    if (synthetic) {
        ds = ann::bench::MakeSynthetic(opt.n, opt.nq, opt.dim, opt.seed);
    } else {
        ds = ann::bench::LoadSIFT(opt.data_dir, opt.max_base);
        if (ds.n == 0) {
            std::fprintf(stderr, "error: failed to load dataset from '%s'\n",
                         opt.data_dir.c_str());
            return 1;
        }
    }

    // ---------------- 构建索引 ----------------
    ann::FlatIndex index(ds.dim, opt.metric);

    const auto build_t0 = std::chrono::steady_clock::now();
    index.Add(ds.n, ds.base.data());
    index.Build();
    const auto build_t1 = std::chrono::steady_clock::now();
    const double build_seconds =
        std::chrono::duration<double>(build_t1 - build_t0).count();

    // ---------------- ground truth ----------------
    std::vector<int32_t> gt = ds.groundtruth;
    size_t gt_dim = ds.gt_dim;
    if (gt.empty()) {
        gt = BuildGroundTruth(&index, ds.queries.data(), ds.nq, opt.k);
        gt_dim = opt.k;
    }

    // ---------------- 预热与测量 ----------------
    ann::bench::LatencyRecorder latency;
    ann::bench::QpsTimer timer;

    for (size_t r = 0; r < opt.warmup; ++r) {
        ann::bench::LatencyRecorder dummy;
        ann::bench::QpsTimer dummy_timer;
        RunRound(index, ds.queries.data(), ds.nq, opt.k, ds.dim, &dummy, &dummy_timer);
    }

    latency.Reset();
    for (size_t r = 0; r < opt.runs; ++r) {
        RunRound(index, ds.queries.data(), ds.nq, opt.k, ds.dim, &latency, &timer);
    }

    // ---------------- 计算指标 ----------------
    // 用最后一轮的返回结果计算召回率。
    std::vector<ann::dist_t> last_dists(ds.nq * opt.k);
    std::vector<ann::idx_t> last_ids(ds.nq * opt.k);
    {
        ann::ResultView out{last_dists.data(), last_ids.data(), opt.k};
        index.SearchBatch(ds.queries.data(), ds.nq, opt.k, out);
    }

    const double recall =
        ann::bench::RecallAtK(last_ids.data(), ds.nq, opt.k, gt.data(), gt_dim);
    const ann::bench::LatencyRecorder::Stats st = latency.Compute();

    // ---------------- 输出 ----------------
    std::printf("\n");
    std::printf("=== ann-engine benchmark ===\n");
    std::printf("index        : %s\n", index.Name());
    std::printf("metric       : %s\n", MetricName(opt.metric));
    std::printf("data source  : %s\n", synthetic ? "synthetic" : opt.data_dir.c_str());
    std::printf("base         : %zu x %zu\n", ds.n, ds.dim);
    std::printf("queries      : %zu\n", ds.nq);
    std::printf("k            : %zu\n", opt.k);
    std::printf("warmup/runs  : %zu/%zu\n", opt.warmup, opt.runs);
    std::printf("\n");
    std::printf("build time   : %.3f s\n", build_seconds);
    std::printf("memory       : %.1f MB\n",
                static_cast<double>(index.MemoryUsage()) / (1024.0 * 1024.0));
    std::printf("recall@%-5zu : %.6f\n", opt.k, recall);
    std::printf("qps          : %.1f\n", timer.Qps());
    std::printf("latency mean : %.4f ms\n", st.mean);
    std::printf("latency p50  : %.4f ms\n", st.p50);
    std::printf("latency p95  : %.4f ms\n", st.p95);
    std::printf("latency p99  : %.4f ms\n", st.p99);
    std::printf("latency max  : %.4f ms\n", st.max);
    std::printf("\n");

    if (synthetic) {
        std::printf("note: synthetic data uses FlatIndex as its own ground truth,\n");
        std::printf("      so recall is 1.0 by construction. Use --data for a real check.\n");
        std::printf("\n");
    }

    return 0;
}
