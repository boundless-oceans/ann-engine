// 对比官方答案、FlatIndex 与独立 double 参考结果，定位召回差异。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include "dataset.h"

#include "ann/flat.h"

namespace {

constexpr size_t k = 10;
using Candidate = std::pair<double, ann::idx_t>;

// 先转换再相减、累加，避免参考计算沿用 float 的中间结果。
double L2Double(const float* a, const float* b, size_t dim) {
    double sum = 0.0;
    for (size_t j = 0; j < dim; ++j) {
        const double diff = static_cast<double>(a[j]) - static_cast<double>(b[j]);
        sum += diff * diff;
    }
    return sum;
}

bool Contains(const ann::idx_t* ids, ann::idx_t id) {
    return std::find(ids, ids + k, id) != ids + k;
}

// 容差仅用于诊断分类，不改变项目 RecallAtK 的严格 ID 匹配规则。
bool SameDistance(double a, double b) {
    return std::abs(a - b) <= 1e-9 * std::max({1.0, std::abs(a), std::abs(b)});
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ann_recall_diag <sift-directory>\n");
        return 1;
    }

    std::fprintf(stderr, "Loading complete dataset from %s...\n", argv[1]);
    const auto ds = ann::bench::LoadSIFT(argv[1]);
    if (ds.n < k || ds.nq == 0 || ds.dim == 0 || ds.gt_dim < k ||
        ds.n >= static_cast<size_t>(ann::kInvalidIdx)) {
        std::fprintf(stderr, "error: dataset is missing, inconsistent or too small for top-10\n");
        return 1;
    }
    for (size_t q = 0; q < ds.nq; ++q) {
        for (size_t i = 0; i < k; ++i) {
            const auto id = ds.groundtruth[q * ds.gt_dim + i];
            if (id < 0 || static_cast<size_t>(id) >= ds.n) {
                std::fprintf(stderr, "error: invalid ground-truth ID at query=%zu\n", q);
                return 1;
            }
            for (size_t j = 0; j < i; ++j) {
                if (id == ds.groundtruth[q * ds.gt_dim + j]) {
                    std::fprintf(stderr, "error: duplicate ground-truth ID at query=%zu\n", q);
                    return 1;
                }
            }
        }
    }

    ann::FlatIndex index(ds.dim, ann::Metric::L2);
    index.Add(ds.n, ds.base.data());
    float dists[k]{};
    ann::idx_t ids[k]{};
    size_t hits = 0;
    size_t mismatches = 0;
    size_t equivalent = 0;
    std::vector<Candidate> reference;
    std::fprintf(stderr, "Searching %zu queries against %zu vectors...\n", ds.nq, ds.n);

    for (size_t q = 0; q < ds.nq; ++q) {
        const float* query = ds.queries.data() + q * ds.dim;
        index.Search(query, k, {dists, ids, k});
        ann::idx_t official[k]{};
        bool valid = true;
        size_t query_hits = 0;
        for (size_t i = 0; i < k; ++i) {
            official[i] = static_cast<ann::idx_t>(ds.groundtruth[q * ds.gt_dim + i]);
            valid = valid && static_cast<size_t>(ids[i]) < ds.n;
            for (size_t j = 0; j < i; ++j) {
                valid = valid && ids[i] != ids[j];
            }
        }
        if (!valid) {
            std::fprintf(stderr, "error: invalid or duplicate result ID at query=%zu\n", q);
            return 1;
        }
        for (const auto id : ids) {
            query_hits += Contains(official, id) ? 1u : 0u;
        }
        hits += query_hits;

        if (query_hits != k) {
            ++mismatches;
            // 仅对不一致查询全量复核；用标准库部分选择，独立于手写最大堆。
            reference.resize(ds.n);
            for (size_t i = 0; i < ds.n; ++i) {
                const double dist = L2Double(query, ds.base.data() + i * ds.dim, ds.dim);
                if (!std::isfinite(dist)) {
                    std::fprintf(stderr, "error: non-finite reference distance\n");
                    return 1;
                }
                reference[i] = {dist, static_cast<ann::idx_t>(i)};
            }
            if (ds.n > k) {
                std::nth_element(reference.begin(), reference.begin() + k, reference.end());
            }
            std::sort(reference.begin(), reference.begin() + k);

            double got_dist[k]{};
            double gt_dist[k]{};
            std::printf("\nquery=%zu hits=%zu/%zu\n", q, query_hits, k);
            std::printf(
                "rank got_id got_float got_double official_id official_double ref_id ref_double\n");
            for (size_t i = 0; i < k; ++i) {
                got_dist[i] = L2Double(query, ds.base.data() + ids[i] * ds.dim, ds.dim);
                gt_dist[i] = L2Double(query, ds.base.data() + official[i] * ds.dim, ds.dim);
                std::printf("%zu %u %.9g %.17g %u %.17g %u %.17g\n",
                            i + 1,
                            static_cast<unsigned>(ids[i]),
                            static_cast<double>(dists[i]),
                            got_dist[i],
                            static_cast<unsigned>(official[i]),
                            gt_dist[i],
                            static_cast<unsigned>(reference[i].second),
                            reference[i].first);
            }
            std::printf("missing_ids:");
            for (const auto id : official) {
                if (!Contains(ids, id)) {
                    std::printf(" %u", static_cast<unsigned>(id));
                }
            }
            std::printf("\nextra_ids:");
            for (const auto id : ids) {
                if (!Contains(official, id)) {
                    std::printf(" %u", static_cast<unsigned>(id));
                }
            }
            std::sort(got_dist, got_dist + k);
            std::sort(gt_dist, gt_dist + k);
            bool got_matches = true;
            bool gt_matches = true;
            for (size_t i = 0; i < k; ++i) {
                got_matches = got_matches && SameDistance(got_dist[i], reference[i].first);
                gt_matches = gt_matches && SameDistance(gt_dist[i], reference[i].first);
            }
            equivalent += got_matches && gt_matches ? 1u : 0u;
            std::printf("\nreference_distance_match: got=%s official=%s (rtol=1e-9, floor=1)\n",
                        got_matches ? "yes" : "no",
                        gt_matches ? "yes" : "no");
            std::fflush(stdout);
        }
        if ((q + 1) % 100 == 0 || q + 1 == ds.nq) {
            std::fprintf(
                stderr, "Progress: %zu/%zu, mismatched queries=%zu\n", q + 1, ds.nq, mismatches);
        }
    }

    std::printf("\n=== recall diagnostic summary ===\n");
    std::printf("base=%zu dim=%zu queries=%zu k=%zu\n", ds.n, ds.dim, ds.nq, k);
    std::printf("strict_recall=%.9f missing_ids=%zu mismatched_queries=%zu\n",
                static_cast<double>(hits) / (static_cast<double>(ds.nq) * k),
                ds.nq * k - hits,
                mismatches);
    std::printf("both_distance_equivalent_to_reference=%zu unresolved_queries=%zu\n",
                equivalent,
                mismatches - equivalent);
    std::printf(
        "note: reference uses distance then ID ordering; it does not establish the official tie "
        "policy.\n");
    return 0;
}
