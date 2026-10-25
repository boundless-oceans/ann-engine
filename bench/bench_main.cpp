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

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dataset.h"
#include "metrics.h"

#include "ann/distance.h"
#include "ann/flat.h"
#include "ann/hnsw.h"
#include "ann/index.h"
#include "ann/ivf.h"
#include "ann/types.h"

namespace {

struct Options {
    std::string index = "flat";
    std::string quantizer = "none";
    ann::HNSWParams hnsw;
    ann::IVFParams ivf;
    std::vector<size_t> nprobe_list;
    bool nprobe_explicit = false;
    bool ivf_explicit = false;
    bool hnsw_explicit = false;
    std::vector<size_t> ef_search_list;
    bool ef_search_explicit = false;
    std::string csv_path;
    size_t n = 100000;  // 合成数据的 base 向量数
    size_t nq = 1000;   // 查询数
    size_t dim = 128;   // 维度
    size_t k = 10;      // top-k
    ann::Metric metric = ann::Metric::L2;
    std::string data_dir;  // 非空则从磁盘加载，忽略合成数据参数
    size_t max_base = 0;   // 截断 base（调试用）
    uint64_t seed = 42;
    size_t warmup = 1;  // 预热轮数
    size_t runs = 3;    // 正式测量轮数
};

void PrintUsage() {
    std::printf(
        "ann_bench - end-to-end benchmark for ann-engine\n"
        "\n"
        "usage: ann_bench [options]\n"
        "\n"
        "options:\n"
        "  --index <name>    flat | hnsw | ivf (default flat)\n"
        "  --m <int>         HNSW M, Mmax0=2*M (default 16)\n"
        "  --ef-construction <int> HNSW build width (default 200)\n"
        "  --ef-search <int> HNSW search width (default 64)\n"
        "  --ef-search-list <ints> HNSW sweep, e.g. 16,32,64,128; build once\n"
        "  --quantizer <name> IVF none | sq8 (default none); sq8 encodes residuals\n"
        "  --nlist <int>     IVF cluster count (default 1024)\n"
        "  --nprobe <int>    IVF requested probe count (default 16)\n"
        "  --nprobe-list <ints> IVF sweep, e.g. 1,4,16; train/build once\n"
        "  --kmeans-iters <int> IVF training iterations (default 25)\n"
        "  --csv <path>      export measurement rows as CSV\n"
        "  --n <int>         number of base vectors for synthetic data (default 100000)\n"
        "  --nq <int>        number of queries (default 1000)\n"
        "  --dim <int>       vector dimension (default 128)\n"
        "  --k <int>         top-k (default 10)\n"
        "  --metric <name>   l2 | ip (default l2)\n"
        "  --data <dir>      load SIFT1M from dir instead of synthetic data\n"
        "  --max-base <int>  truncate base vectors (debug only)\n"
        "  --seed <int>      random seed for data and index training/build (default 42)\n"
        "  --warmup <int>    warmup rounds, not measured (default 1)\n"
        "  --runs <int>      measured rounds (default 3)\n"
        "  -h, --help        show this message\n");
}

// 严格解析正整数列表；保留输入顺序与重复项，支持反向/重复测量观察顺序影响。
bool ParsePositiveList(std::string_view text, std::vector<size_t>* values) {
    values->clear();
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto token = text.substr(0, comma);
        size_t value = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        if (token.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != token.data() + token.size() || value == 0) {
            return false;
        }
        values->push_back(value);
        if (comma == std::string_view::npos) {
            return true;
        }
        text.remove_prefix(comma + 1);
    }
    return false;
}

bool ParsePositive(std::string_view text, size_t* value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), *value);
    return !text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
           *value != 0;
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
        } else if (a == "--index") {
            const char* v = need_value("--index");
            if (v == nullptr)
                return false;
            opt->index = v;
        } else if (a == "--quantizer") {
            const char* v = need_value("--quantizer");
            if (v == nullptr)
                return false;
            opt->quantizer = v;
            opt->ivf_explicit = true;
            if (opt->quantizer != "none" && opt->quantizer != "sq8") {
                std::fprintf(stderr, "error: quantizer must be none or sq8\n");
                return false;
            }
        } else if (a == "--m") {
            opt->hnsw_explicit = true;
            const char* v = need_value("--m");
            if (v == nullptr)
                return false;
            opt->hnsw.M = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--ef-construction") {
            opt->hnsw_explicit = true;
            const char* v = need_value("--ef-construction");
            if (v == nullptr)
                return false;
            opt->hnsw.ef_construction = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--ef-search") {
            opt->hnsw_explicit = true;
            const char* v = need_value("--ef-search");
            if (v == nullptr)
                return false;
            opt->hnsw.ef_search = static_cast<size_t>(std::strtoull(v, nullptr, 10));
            opt->ef_search_explicit = true;
        } else if (a == "--ef-search-list") {
            opt->hnsw_explicit = true;
            const char* v = need_value("--ef-search-list");
            if (v == nullptr)
                return false;
            if (!ParsePositiveList(v, &opt->ef_search_list)) {
                std::fprintf(stderr,
                             "error: ef-search-list requires positive comma-separated integers\n");
                return false;
            }
        } else if (a == "--nlist" || a == "--nprobe" || a == "--kmeans-iters") {
            const char* v = need_value(a.c_str());
            if (v == nullptr)
                return false;
            size_t* target = a == "--nlist"    ? &opt->ivf.nlist
                             : a == "--nprobe" ? &opt->ivf.nprobe
                                               : &opt->ivf.kmeans_iters;
            if (!ParsePositive(v, target)) {
                std::fprintf(stderr, "error: %s requires a positive integer\n", a.c_str());
                return false;
            }
            opt->ivf_explicit = true;
            opt->nprobe_explicit |= a == "--nprobe";
        } else if (a == "--nprobe-list") {
            const char* v = need_value("--nprobe-list");
            if (v == nullptr)
                return false;
            if (!ParsePositiveList(v, &opt->nprobe_list)) {
                std::fprintf(stderr,
                             "error: nprobe-list requires positive comma-separated integers\n");
                return false;
            }
            opt->ivf_explicit = true;
        } else if (a == "--csv") {
            const char* v = need_value("--csv");
            if (v == nullptr)
                return false;
            opt->csv_path = v;
        } else if (a == "--n") {
            const char* v = need_value("--n");
            if (v == nullptr)
                return false;
            opt->n = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--nq") {
            const char* v = need_value("--nq");
            if (v == nullptr)
                return false;
            opt->nq = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--dim") {
            const char* v = need_value("--dim");
            if (v == nullptr)
                return false;
            opt->dim = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--k") {
            const char* v = need_value("--k");
            if (v == nullptr)
                return false;
            opt->k = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--metric") {
            const char* v = need_value("--metric");
            if (v == nullptr)
                return false;
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
            if (v == nullptr)
                return false;
            opt->data_dir = v;
        } else if (a == "--max-base") {
            const char* v = need_value("--max-base");
            if (v == nullptr)
                return false;
            opt->max_base = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--seed") {
            const char* v = need_value("--seed");
            if (v == nullptr)
                return false;
            opt->seed = std::strtoull(v, nullptr, 10);
        } else if (a == "--warmup") {
            const char* v = need_value("--warmup");
            if (v == nullptr)
                return false;
            opt->warmup = static_cast<size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--runs") {
            const char* v = need_value("--runs");
            if (v == nullptr)
                return false;
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
    if (opt->index != "flat" && opt->index != "hnsw" && opt->index != "ivf") {
        std::fprintf(stderr, "error: index must be flat, hnsw or ivf\n");
        return false;
    }
    if (!opt->ef_search_list.empty() && (opt->index != "hnsw" || opt->ef_search_explicit)) {
        std::fprintf(stderr,
                     "error: ef-search-list requires hnsw and cannot combine with ef-search\n");
        return false;
    }
    if (opt->index == "hnsw" && (opt->hnsw.M < 2 || opt->hnsw.M > ann::kInvalidIdx / 2 ||
                                 opt->hnsw.ef_construction == 0 || opt->hnsw.ef_search == 0)) {
        std::fprintf(stderr, "error: invalid HNSW parameters\n");
        return false;
    }
    if (opt->ivf_explicit && opt->index != "ivf") {
        std::fprintf(stderr, "error: IVF options require --index ivf\n");
        return false;
    }
    if (opt->index == "ivf" && (opt->metric != ann::Metric::L2 || opt->hnsw_explicit ||
                                opt->ivf.nlist > ann::kInvalidIdx ||
                                (!opt->nprobe_list.empty() && opt->nprobe_explicit))) {
        std::fprintf(stderr,
                     "error: IVF requires l2, IVF-only parameters and mutually exclusive "
                     "nprobe/nprobe-list\n");
        return false;
    }
    if (!opt->data_dir.empty() && opt->metric != ann::Metric::L2) {
        std::fprintf(stderr, "error: SIFT official ground truth requires l2\n");
        return false;
    }
    if (opt->index == "hnsw") {
        opt->hnsw.Mmax0 = 2 * opt->hnsw.M;
        opt->hnsw.seed = opt->seed;
    }
    opt->ivf.seed = opt->seed;
    return true;
}

const char* MetricName(ann::Metric m) {
    switch (m) {
        case ann::Metric::L2:
            return "L2";
        case ann::Metric::InnerProduct:
            return "InnerProduct";
        case ann::Metric::Cosine:
            return "Cosine";
    }
    return "Unknown";
}

// 用精确 Flat 生成参考；被测对象也是 Flat 时仅作为链路自检。
std::vector<int32_t> BuildGroundTruth(ann::Index* idx, const float* queries, size_t nq, size_t k) {
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
void RunRound(const ann::Index& idx,
              const float* queries,
              size_t nq,
              size_t k,
              size_t dim,
              ann::bench::LatencyRecorder* latency,
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

int RunBenchmark(int argc, char** argv) {
    Options opt;
    if (!ParseArgs(argc, argv, &opt)) {
        return 1;
    }

    std::ofstream csv;
    if (!opt.csv_path.empty()) {
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv.open(opt.csv_path);
        csv << "index,metric,n,dim,nq,k,M,Mmax0,ef_construction,ef_search,effective_ef_search,"
               "seed,warmup,runs,build_seconds,memory_bytes,recall,qps,mean_ms,p50_ms,p95_ms,p99_"
               "ms,max_ms,nlist,nprobe,effective_nprobe,kmeans_iters,quantizer,encoding,code_size,"
               "lut_floats\n";
        csv << std::fixed << std::setprecision(9);
        csv.flush();
    }
    std::fprintf(stderr, "[bench] Loading dataset\n");
    // ---------------- 准备数据 ----------------
    ann::bench::Dataset ds;
    bool synthetic = opt.data_dir.empty();

    if (synthetic) {
        ds = ann::bench::MakeSynthetic(opt.n, opt.nq, opt.dim, opt.seed);
    } else {
        ds = ann::bench::LoadSIFT(opt.data_dir, opt.max_base);
        if (ds.n == 0) {
            std::fprintf(stderr, "error: failed to load dataset from '%s'\n", opt.data_dir.c_str());
            return 1;
        }
    }

    if (opt.index == "ivf" && opt.ivf.nlist > ds.n) {
        std::fprintf(stderr, "error: nlist must not exceed the loaded base vector count\n");
        return 1;
    }

    // ---------------- 构建索引 ----------------
    std::unique_ptr<ann::Index> owned_index;
    if (opt.index == "hnsw") {
        owned_index = std::make_unique<ann::HNSWIndex>(ds.dim, opt.metric, opt.hnsw);
    } else if (opt.index == "ivf") {
        std::unique_ptr<ann::Quantizer> quantizer;
        if (opt.quantizer == "sq8") {
            quantizer = std::make_unique<ann::ScalarQuantizer>(ds.dim);
        }
        owned_index =
            std::make_unique<ann::IVFIndex>(ds.dim, opt.metric, opt.ivf, std::move(quantizer));
    } else {
        owned_index = std::make_unique<ann::FlatIndex>(ds.dim, opt.metric);
    }
    ann::Index& index = *owned_index;

    std::fprintf(stderr, "[bench] Building %s: %zu vectors\n", index.Name(), ds.n);
    const auto build_t0 = std::chrono::steady_clock::now();
    index.Add(ds.n, ds.base.data());
    index.Build();
    const auto build_t1 = std::chrono::steady_clock::now();
    const double build_seconds = std::chrono::duration<double>(build_t1 - build_t0).count();

    std::fprintf(stderr, "[bench] Build completed in %.3f s\n", build_seconds);
    // ---------------- ground truth ----------------
    std::vector<int32_t> gt = ds.groundtruth;
    size_t gt_dim = ds.gt_dim;
    if (gt.empty()) {
        std::fprintf(stderr, "[bench] Computing exact Flat reference\n");
        if (opt.index != "flat") {
            // ANN 不得把自身输出当作答案；独立 Flat 枚举产生严格 ID 参考。
            ann::FlatIndex reference(ds.dim, opt.metric);
            reference.Add(ds.n, ds.base.data());
            gt = BuildGroundTruth(&reference, ds.queries.data(), ds.nq, opt.k);
        } else {
            gt = BuildGroundTruth(&index, ds.queries.data(), ds.nq, opt.k);
        }
        gt_dim = opt.k;
    }

    const std::vector<size_t> search_values =
        opt.index == "ivf"
            ? (opt.nprobe_list.empty() ? std::vector<size_t>{opt.ivf.nprobe} : opt.nprobe_list)
            : (opt.ef_search_list.empty() ? std::vector<size_t>{opt.hnsw.ef_search}
                                          : opt.ef_search_list);
    for (const size_t value : search_values) {
        if (opt.index == "hnsw") {
            static_cast<ann::HNSWIndex&>(index).SetEfSearch(value);
        } else if (opt.index == "ivf") {
            static_cast<ann::IVFIndex&>(index).SetNProbe(value);
        }
        std::fprintf(stderr,
                     "[bench] Measuring %s: ef=%zu, nprobe=%zu, warmup=%zu, runs=%zu\n",
                     index.Name(),
                     opt.index == "hnsw" ? value : 0,
                     opt.index == "ivf" ? value : 0,
                     opt.warmup,
                     opt.runs);
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
        // 额外执行一轮只统计召回，不计入正式查询时间。
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
        const auto& cpu = ann::CpuFeatures::Get();
        const auto dist_fn = ann::SelectDistFn(opt.metric);
        const bool simd = dist_fn == ann::L2SqrAVX2Four || dist_fn == ann::InnerProductAVX2Four;
        std::printf("cpu avx2/fma : %s/%s\n", cpu.avx2 ? "yes" : "no", cpu.fma ? "yes" : "no");
        if (opt.quantizer == "sq8") {
            std::printf("coarse fn    : %s\n", simd ? "AVX2Four" : "Scalar");
            std::printf("candidate fn : SQ8 ADC (scalar)\n");
        } else {
            std::printf("distance fn  : %s\n", simd ? "AVX2Four" : "Scalar");
        }
        std::printf("data source  : %s\n", synthetic ? "synthetic" : opt.data_dir.c_str());
        std::printf("base         : %zu x %zu\n", ds.n, ds.dim);
        std::printf("queries      : %zu\n", ds.nq);
        std::printf("k            : %zu\n", opt.k);
        std::printf("warmup/runs  : %zu/%zu\n", opt.warmup, opt.runs);
        if (opt.index == "hnsw") {
            const auto& hnsw = static_cast<const ann::HNSWIndex&>(index);
            const auto stats = hnsw.GetStats();
            std::printf("M/Mmax0      : %zu/%zu\n", hnsw.Params().M, hnsw.Params().Mmax0);
            std::printf(
                "ef build/run : %zu/%zu\n", hnsw.Params().ef_construction, hnsw.GetEfSearch());
            std::printf("max level    : %d\n", stats.max_level);
            std::printf("avg degree0  : %.2f\n", stats.avg_degree_l0);
        }
        if (opt.index == "ivf") {
            std::printf("nlist        : %zu\n", opt.ivf.nlist);
            std::printf("nprobe req/eff: %zu/%zu\n", value, std::min(value, opt.ivf.nlist));
            std::printf("kmeans iters : %zu\n", opt.ivf.kmeans_iters);
            std::printf("quantizer    : %s\n", opt.quantizer.c_str());
            std::printf("encoding     : %s\n", opt.quantizer == "sq8" ? "residual" : "raw");
            if (opt.quantizer == "sq8") {
                std::printf("code bytes   : %zu per vector\n", ds.dim);
                std::printf("LUT floats   : %zu per query, rebuilt for each nonempty list\n",
                            ds.dim * 256);
            }
        }
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

        if (synthetic && opt.index != "flat") {
            std::printf(
                "note: synthetic ground truth was computed independently with FlatIndex.\n");
        } else if (synthetic) {
            std::printf("note: synthetic data uses FlatIndex as its own ground truth,\n");
            std::printf("      so recall is 1.0 by construction. Use --data for a real check.\n");
            std::printf("\n");
        }

        if (csv.is_open()) {
            const auto& params = opt.hnsw;
            const bool hnsw = opt.index == "hnsw";
            csv << index.Name() << ',' << MetricName(opt.metric) << ',' << ds.n << ',' << ds.dim
                << ',' << ds.nq << ',' << opt.k << ',' << (hnsw ? params.M : 0) << ','
                << (hnsw ? params.Mmax0 : 0) << ',' << (hnsw ? params.ef_construction : 0) << ','
                << (hnsw ? value : 0) << ',' << (hnsw ? std::max(value, opt.k) : 0) << ','
                << opt.seed << ',' << opt.warmup << ',' << opt.runs << ',' << build_seconds << ','
                << index.MemoryUsage() << ',' << recall << ',' << timer.Qps() << ',' << st.mean
                << ',' << st.p50 << ',' << st.p95 << ',' << st.p99 << ',' << st.max << ','
                << (opt.index == "ivf" ? opt.ivf.nlist : 0) << ','
                << (opt.index == "ivf" ? value : 0) << ','
                << (opt.index == "ivf" ? std::min(value, opt.ivf.nlist) : 0) << ','
                << (opt.index == "ivf" ? opt.ivf.kmeans_iters : 0) << ',' << opt.quantizer << ','
                << (opt.quantizer == "sq8" ? "residual" : "raw") << ','
                << (opt.quantizer == "sq8" ? ds.dim : 0) << ','
                << (opt.quantizer == "sq8" ? ds.dim * 256 : 0) << '\n';
            csv.flush();
        }
        std::fflush(stdout);
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        return RunBenchmark(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
