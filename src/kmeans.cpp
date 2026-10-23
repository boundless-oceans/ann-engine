#include "ann/kmeans.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "kmeans_internal.h"

namespace ann {
namespace {

size_t Elements(size_t n, size_t dim) {
    if (n > std::numeric_limits<size_t>::max() / dim / sizeof(float)) {
        throw std::length_error("KMeans: matrix byte size overflow");
    }
    return n * dim;
}

void CheckFinite(const float* data, size_t count) {
    if (data == nullptr) {
        throw std::invalid_argument("KMeans: null matrix");
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument("KMeans: non-finite coordinate");
        }
    }
}

// 基线先使用 double 距离与累加，避免有限 float 在平方/求和时溢出。
double Distance(const float* a, const float* b, size_t dim) {
    double result = 0;
    for (size_t d = 0; d < dim; ++d) {
        const double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        result += diff * diff;
    }
    return result;
}

size_t Nearest(const float* point, const float* centers, size_t k, size_t dim) {
    size_t best = 0;
    double best_distance = Distance(point, centers, dim);
    for (size_t c = 1; c < k; ++c) {
        const double distance = Distance(point, centers + c * dim, dim);
        if (distance < best_distance) {
            best = c;
            best_distance = distance;
        }
    }
    return best;
}

}  // namespace

namespace detail {

void ReseedEmptyKMeansClusters(size_t n,
                               size_t dim,
                               const float* data,
                               std::span<float> centers,
                               std::span<const size_t> counts,
                               std::span<double> nearest,
                               std::span<unsigned char> chosen) {
    // 空簇重置：相对非空中心取最远样本；后续空簇也考虑刚重置的中心。
    std::fill(nearest.begin(), nearest.end(), std::numeric_limits<double>::infinity());
    std::fill(chosen.begin(), chosen.end(), 0);
    for (size_t c = 0; c < counts.size(); ++c) {
        if (counts[c] != 0) {
            for (size_t i = 0; i < n; ++i) {
                nearest[i] =
                    std::min(nearest[i], Distance(data + i * dim, centers.data() + c * dim, dim));
            }
        }
    }
    for (size_t c = 0; c < counts.size(); ++c) {
        if (counts[c] != 0) {
            continue;
        }
        size_t farthest = n;
        for (size_t i = 0; i < n; ++i) {
            if (chosen[i] == 0 && (farthest == n || nearest[i] > nearest[farthest])) {
                farthest = i;
            }
        }
        chosen[farthest] = 1;
        std::copy_n(data + farthest * dim, dim, centers.data() + c * dim);
        for (size_t i = 0; i < n; ++i) {
            nearest[i] =
                std::min(nearest[i], Distance(data + i * dim, centers.data() + c * dim, dim));
        }
    }
}

}  // namespace detail

KMeans::KMeans(size_t dim, size_t k, size_t iters, uint64_t seed)
    : dim_(dim), k_(k), iters_(iters), rng_(seed) {
    if (dim == 0 || k == 0 || iters == 0 || k > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("KMeans: invalid dimension, cluster count or iterations");
    }
    (void)Elements(k, dim);
}

void KMeans::Assign(size_t n,
                    const float* data,
                    const float* centroids,
                    uint32_t* out_assign) const {
    if (n == 0) {
        return;
    }
    CheckFinite(data, Elements(n, dim_));
    CheckFinite(centroids, Elements(k_, dim_));
    if (out_assign == nullptr) {
        throw std::invalid_argument("KMeans: null assignments");
    }
    for (size_t i = 0; i < n; ++i) {
        out_assign[i] = static_cast<uint32_t>(Nearest(data + i * dim_, centroids, k_, dim_));
    }
}

void KMeans::Train(size_t n, const float* data, float* out_centroids) {
    if (n < k_ || out_centroids == nullptr) {
        throw std::invalid_argument("KMeans: training requires n>=k and output storage");
    }
    CheckFinite(data, Elements(n, dim_));
    const size_t elements = Elements(k_, dim_);
    std::vector<float> centers(elements);
    std::vector<double> nearest(n, std::numeric_limits<double>::infinity());
    std::vector<unsigned char> chosen(n, 0);
    auto rng = rng_;

    // k-means++ 按到已选中心的最近平方距离抽样；全零权重时选未用样本。
    size_t selected = std::uniform_int_distribution<size_t>(0, n - 1)(rng);
    for (size_t c = 0; c < k_; ++c) {
        std::copy_n(data + selected * dim_, dim_, centers.data() + c * dim_);
        chosen[selected] = 1;
        double total = 0;
        for (size_t i = 0; i < n; ++i) {
            nearest[i] =
                std::min(nearest[i], Distance(data + i * dim_, centers.data() + c * dim_, dim_));
            total += nearest[i];
        }
        if (c + 1 == k_) {
            break;
        }
        selected = 0;
        if (total > 0) {
            const double target = std::generate_canonical<double, 53>(rng) * total;
            double cumulative = 0;
            for (size_t i = 0; i < n; ++i) {
                if (nearest[i] > 0) {
                    selected = i;  // 浮点求和边界回退到最后一个正权重点。
                    cumulative += nearest[i];
                    if (target < cumulative) {
                        break;
                    }
                }
            }
        } else {
            while (chosen[selected] != 0) {
                ++selected;
            }
        }
    }

    std::vector<double> sums(elements);
    std::vector<size_t> counts(k_);
    for (size_t iteration = 0; iteration < iters_; ++iteration) {
        std::fill(sums.begin(), sums.end(), 0);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            const size_t c = Nearest(data + i * dim_, centers.data(), k_, dim_);
            ++counts[c];
            for (size_t d = 0; d < dim_; ++d) {
                sums[c * dim_ + d] += static_cast<double>(data[i * dim_ + d]);
            }
        }
        for (size_t c = 0; c < k_; ++c) {
            if (counts[c] != 0) {
                for (size_t d = 0; d < dim_; ++d) {
                    centers[c * dim_ + d] =
                        static_cast<float>(sums[c * dim_ + d] / static_cast<double>(counts[c]));
                }
            }
        }
        if (std::find(counts.begin(), counts.end(), size_t{0}) == counts.end()) {
            continue;
        }
        detail::ReseedEmptyKMeansClusters(n, dim_, data, centers, counts, nearest, chosen);
    }
    // 最后一次更新改变了质心，统计必须对应实际交付的质心，而非上一轮分配。
    std::fill(counts.begin(), counts.end(), 0);
    for (size_t i = 0; i < n; ++i) {
        ++counts[Nearest(data + i * dim_, centers.data(), k_, dim_)];
    }
    std::copy(centers.begin(), centers.end(), out_centroids);
    cluster_sizes_.swap(counts);
    rng_ = rng;
}

}  // namespace ann
