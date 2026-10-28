#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "dataset.h"

#include "ann/flat.h"
#include "ann/ivf.h"
#include "ann/kmeans.h"

namespace {

size_t Positive(std::string_view text) {
    size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        throw std::invalid_argument("Expected a positive integer");
    }
    return value;
}

void Residual(const float* x, const float* center, size_t dim, float* out) {
    for (size_t d = 0; d < dim; ++d) {
        out[d] = static_cast<float>(static_cast<double>(x[d]) - static_cast<double>(center[d]));
    }
}

double L2(const float* a, const float* b, size_t dim) {
    double sum = 0;
    for (size_t d = 0; d < dim; ++d) {
        const double delta = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        sum += delta * delta;
    }
    return sum;
}

// 分解路径复用训练/编码实现，但独立于IVF的列表、码寻址、查询和堆逻辑。
// double解码距离不查表；它检验ADC数值关系，不证明训练器本身正确。
bool Diagnose(const ann::bench::Dataset& data,
              size_t nlist,
              size_t m,
              size_t ksub,
              size_t k,
              size_t coarse_iters) {
    const size_t n = data.n, dim = data.dim, nq = data.nq;
    ann::IVFParams params;
    params.nlist = params.nprobe = nlist;
    params.kmeans_iters = coarse_iters;
    params.seed = 42;
    std::cerr << "Training decomposed reference\n";
    ann::KMeans coarse(dim, nlist, params.kmeans_iters, params.seed);
    std::vector<float> centers(nlist * dim), residuals(n * dim), decoded(n * dim);
    std::vector<uint32_t> assignment(n);
    coarse.Train(n, data.base.data(), centers.data());
    coarse.Assign(n, data.base.data(), centers.data(), assignment.data());
    for (size_t id = 0; id < n; ++id) {
        Residual(data.base.data() + id * dim,
                 centers.data() + size_t{assignment[id]} * dim,
                 dim,
                 residuals.data() + id * dim);
    }
    ann::ProductQuantizer pq(dim, m, ksub, 42);
    pq.Train(n, residuals.data());
    std::vector<uint8_t> codes(n * m);
    std::vector<double> squared_error(dim, 0);
    double residual_energy = 0;
    for (size_t id = 0; id < n; ++id) {
        pq.Encode(residuals.data() + id * dim, codes.data() + id * m);
        pq.Decode(codes.data() + id * m, decoded.data() + id * dim);
        for (size_t d = 0; d < dim; ++d) {
            const double r = residuals[id * dim + d];
            const double error = static_cast<double>(decoded[id * dim + d]) - r;
            squared_error[d] += error * error;
            residual_energy += r * r;
        }
    }

    std::cerr << "Building IVF and exact reference\n";
    ann::IVFIndex index(
        dim, ann::Metric::L2, params, std::make_unique<ann::ProductQuantizer>(dim, m, ksub, 42));
    index.Add(n, data.base.data());
    index.Build();
    ann::FlatIndex flat(dim, ann::Metric::L2);
    flat.Add(n, data.base.data());
    std::vector<std::vector<size_t>> lists(nlist);
    for (size_t id = 0; id < n; ++id) {
        lists[assignment[id]].push_back(id);
    }
    std::vector<float> query_residual(dim), lut(pq.LutSize()), distances(k), exact_distances(k);
    std::vector<ann::idx_t> ids(k), exact_ids(k);
    std::vector<std::pair<float, ann::idx_t>> adc(n);
    std::vector<std::pair<double, ann::idx_t>> decoded_order(n);
    std::vector<size_t> ranks(n);
    size_t ivf_id_mismatches = 0, ivf_distance_mismatches = 0, decoded_id_mismatches = 0;
    size_t hits = 0, decoded_hits = 0, lost = 0, worst_rank = 0;
    double max_relative_adc_error = 0, lost_rank_sum = 0, lost_distance_bias = 0;
    std::cout << std::setprecision(12) << "base=" << n << " dim=" << dim << " queries=" << nq
              << " k=" << k << " nlist=" << nlist << " nprobe=" << nlist << " pq_m=" << m
              << " ksub=" << ksub << " coarse_iters=" << coarse_iters << " pq_iters=25 seed=42\n"
              << "reference=FlatIndex-on-subset; no QPS; no reranking\n";
    std::cerr << "Comparing exhaustive ADC, decoded double distances and IVF results\n";
    size_t examples = 0;
    for (size_t q = 0; q < nq; ++q) {
        const float* query = data.queries.data() + q * dim;
        index.Search(query, k, {distances.data(), ids.data(), k});
        flat.Search(query, k, {exact_distances.data(), exact_ids.data(), k});
        for (size_t c = 0; c < nlist; ++c) {
            if (lists[c].empty()) {
                continue;
            }
            Residual(query, centers.data() + c * dim, dim, query_residual.data());
            pq.BuildLUT(query_residual.data(), lut.data());
            for (const size_t id : lists[c]) {
                const float approximate = pq.DistanceFromLUT(lut.data(), codes.data() + id * m);
                const double reference = L2(query_residual.data(), decoded.data() + id * dim, dim);
                const double relative = std::abs(static_cast<double>(approximate) - reference) /
                                        std::max(1.0, reference);
                max_relative_adc_error = std::max(max_relative_adc_error, relative);
                adc[id] = {approximate, static_cast<ann::idx_t>(id)};
                decoded_order[id] = {reference, static_cast<ann::idx_t>(id)};
            }
        }
        std::sort(adc.begin(), adc.end());
        std::sort(decoded_order.begin(), decoded_order.end());
        for (size_t rank = 0; rank < n; ++rank) {
            ranks[adc[rank].second] = rank + 1;
        }
        for (size_t rank = 0; rank < k; ++rank) {
            ivf_id_mismatches += ids[rank] != adc[rank].second;
            // 同一float计算路径应逐位一致，double参考另设容差。
            ivf_distance_mismatches += std::bit_cast<uint32_t>(distances[rank]) !=
                                       std::bit_cast<uint32_t>(adc[rank].first);
            decoded_id_mismatches += adc[rank].second != decoded_order[rank].second;
            const auto found = std::find(exact_ids.begin(), exact_ids.end(), ids[rank]);
            hits += found != exact_ids.end();
            decoded_hits +=
                std::find(exact_ids.begin(), exact_ids.end(), decoded_order[rank].second) !=
                exact_ids.end();
            const size_t id = exact_ids[rank];
            if (ranks[id] > k) {
                ++lost;
                worst_rank = std::max(worst_rank, ranks[id]);
                lost_rank_sum += static_cast<double>(ranks[id]);
                const double exact = L2(query, data.base.data() + id * dim, dim);
                const double approx = adc[ranks[id] - 1].first;
                lost_distance_bias += approx - exact;
                if (examples++ < 10) {
                    std::cout << "lost query=" << q << " id=" << id << " exact_rank=" << rank + 1
                              << " adc_rank=" << ranks[id] << " exact_d2=" << exact
                              << " adc_d2=" << approx << " adc_topk_boundary=" << adc[k - 1].first
                              << '\n';
                }
            }
        }
    }
    const double total = static_cast<double>(nq * k);
    const double error_sum = std::accumulate(squared_error.begin(), squared_error.end(), 0.0);
    std::cout << "ivf_vs_exhaustive_adc_id_mismatches=" << ivf_id_mismatches
              << " distance_mismatches=" << ivf_distance_mismatches << '\n'
              << "adc_vs_decoded_double_topk_position_mismatches=" << decoded_id_mismatches << '\n'
              << "max_relative_adc_distance_error=" << max_relative_adc_error << '\n'
              << "strict_recall=" << static_cast<double>(hits) / total
              << " decoded_double_recall=" << static_cast<double>(decoded_hits) / total << '\n'
              << "reconstruction_mse_per_coordinate=" << error_sum / static_cast<double>(n * dim)
              << " residual_relative_squared_error="
              << (residual_energy > 0 ? error_sum / residual_energy : 0) << '\n'
              << "coarse_residual_mse_per_coordinate="
              << residual_energy / static_cast<double>(n * dim)
              << " index_capacity_bytes=" << index.MemoryUsage() << '\n'
              << "lost_exact_neighbors=" << lost
              << " mean_adc_rank=" << (lost > 0 ? lost_rank_sum / static_cast<double>(lost) : 0)
              << " worst_adc_rank=" << worst_rank << " mean_distance_bias="
              << (lost > 0 ? lost_distance_bias / static_cast<double>(lost) : 0) << '\n'
              << "subspace,mse_per_coordinate\n";
    const size_t dsub = dim / m;
    for (size_t s = 0; s < m; ++s) {
        const double error = std::accumulate(
            squared_error.begin() + s * dsub, squared_error.begin() + (s + 1) * dsub, 0.0);
        std::cout << s << ',' << error / static_cast<double>(n * dsub) << '\n';
    }
    // double路径可因float LUT平局而换序，单独报告，不当作接入失败。
    return ivf_id_mismatches == 0 && ivf_distance_mismatches == 0 && max_relative_adc_error < 2e-6;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if ((argc == 2 || argc == 3) && std::string_view(argv[1]) == "--self-check") {
            const size_t coarse_iters = argc == 3 ? Positive(argv[2]) : 5;
            ann::bench::Dataset data;
            data.n = 4;
            data.dim = 4;
            data.nq = 2;
            data.base = {0, 0, 10, 10, 0, 0, 12, 12, 2, 2, 10, 10, 2, 2, 12, 12};
            data.queries = {3, 3, 13, 13, 0, 0, 10, 10};
            return Diagnose(data, 1, 2, 2, 2, coarse_iters) ? 0 : 1;
        }
        if (argc < 2 || argc > 5) {
            throw std::invalid_argument(
                "Usage: ann_pq_diag DATA_DIR [nlist=256] [pq_m=32] [coarse_iters=5]");
        }
        const size_t nlist = argc > 2 ? Positive(argv[2]) : 256;
        const size_t m = argc > 3 ? Positive(argv[3]) : 32;
        const size_t coarse_iters = argc > 4 ? Positive(argv[4]) : 5;
        auto data = ann::bench::LoadSIFT(argv[1], 10000);
        if (data.n != 10000 || data.nq < 100 || nlist > data.n || m > data.dim ||
            data.dim % m != 0) {
            throw std::invalid_argument("Requires 10000 base rows, 100 queries and valid PQ shape");
        }
        // 固定诊断窗口，全部参考重算；不使用百万库的官方答案。
        data.nq = 100;
        data.queries.resize(data.nq * data.dim);
        data.groundtruth.clear();
        return Diagnose(data, nlist, m, 256, 10, coarse_iters) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
