#include "ann/ivf.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "ivf_lists.h"

#include "ann/kmeans.h"

namespace ann {
namespace {

void CheckFinite(const float* values, size_t count) {
    if (values == nullptr) {
        throw std::invalid_argument("IVF requires a non-null vector buffer");
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            throw std::invalid_argument("IVF requires finite coordinates");
        }
    }
}

dist_t CheckedDistance(DistFn distance, const float* a, const float* b, size_t dim) {
    const auto value = distance(a, b, dim);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("IVF requires finite distances");
    }
    return value;
}

}  // namespace

IVFIndex::IVFIndex(size_t dim,
                   Metric metric,
                   IVFParams params,
                   std::unique_ptr<Quantizer> quantizer)
    : dim_(dim),
      metric_(metric),
      params_(params),
      dist_fn_(SelectDistFn(metric)),
      nprobe_(params.nprobe) {
    if (dim == 0 || params.nlist == 0 || params.nlist >= std::numeric_limits<size_t>::max() ||
        params.nlist > static_cast<size_t>(kInvalidIdx) || params.nprobe == 0 ||
        params.kmeans_iters == 0) {
        throw std::invalid_argument("IVF requires positive representable parameters");
    }
    if (metric != Metric::L2) {
        throw std::invalid_argument("IVFFlat currently supports L2 only");
    }
    if (quantizer) {
        throw std::invalid_argument("IVFFlat does not support quantizers yet");
    }
    if (dim > std::numeric_limits<size_t>::max() / sizeof(float) / params.nlist) {
        throw std::length_error("IVF centroid storage size overflow");
    }
}

size_t IVFIndex::Size() const {
    return raw_vectors_.size() / dim_;
}

void IVFIndex::Add(size_t n, const float* vecs) {
    if (built_) {
        throw std::logic_error("IVF cannot Add after Build");
    }
    if (n == 0) {
        return;
    }
    if (n > static_cast<size_t>(kInvalidIdx) - Size() ||
        n > (raw_vectors_.max_size() - raw_vectors_.size()) / dim_) {
        throw std::length_error("IVF vector storage exceeds available IDs or capacity");
    }
    const size_t elements = n * dim_;
    CheckFinite(vecs, elements);
    // 原始存储没有公开可变视图，外部输入不会合法别名此私有数组。
    raw_vectors_.insert(raw_vectors_.end(), vecs, vecs + elements);
}

void IVFIndex::Build() {
    if (built_) {
        return;
    }
    const size_t n = Size();
    if (n < params_.nlist) {
        throw std::invalid_argument("IVF Build requires at least nlist training vectors");
    }
    // 全部结果先在局部构建；训练或分配失败时保留未构建状态和已 Add 数据。
    std::vector<float> centers(params_.nlist * dim_);
    std::vector<uint32_t> assignments(n);
    KMeans trainer(dim_, params_.nlist, params_.kmeans_iters, params_.seed);
    trainer.Train(n, raw_vectors_.data(), centers.data());
    trainer.Assign(n, raw_vectors_.data(), centers.data(), assignments.data());
    auto lists = detail::BuildIVFLists(assignments, params_.nlist);
    centroids_.swap(centers);
    list_offsets_.swap(lists.offsets);
    list_ids_.swap(lists.ids);
    built_ = true;
}

void IVFIndex::Search(const float* query, size_t k, ResultView out) const {
    const size_t want = std::min(k, out.k);
    if (want == 0 || query == nullptr || out.ids == nullptr || out.dists == nullptr) {
        return;
    }
    if (!built_) {
        throw std::logic_error("IVF Search requires Build");
    }
    CheckFinite(query, dim_);
    const size_t probes = std::min(GetNProbe(), params_.nlist);
    // 距离、簇 ID 构成全序；只排序需要探测的前缀。
    using Candidate = std::pair<dist_t, idx_t>;
    std::vector<Candidate> clusters;
    if (probes != params_.nlist) {
        clusters.reserve(params_.nlist);
        for (size_t c = 0; c < params_.nlist; ++c) {
            clusters.emplace_back(
                CheckedDistance(dist_fn_, query, centroids_.data() + c * dim_, dim_),
                static_cast<idx_t>(c));
        }
        std::partial_sort(clusters.begin(), clusters.begin() + probes, clusters.end());
    }
    // 局部最大堆保存 top-k，平局按较小原始 ID；不改变 P0 的严格 Recall 定义。
    std::vector<Candidate> results;
    const size_t limit = std::min(want, Size());
    results.reserve(limit);
    for (size_t p = 0; p < probes; ++p) {
        const size_t c = probes == params_.nlist ? p : clusters[p].second;
        for (size_t pos = list_offsets_[c]; pos < list_offsets_[c + 1]; ++pos) {
            const idx_t id = list_ids_[pos];
            const Candidate candidate{
                CheckedDistance(dist_fn_, query, raw_vectors_.data() + size_t{id} * dim_, dim_),
                id};
            if (results.size() < limit) {
                results.push_back(candidate);
                std::push_heap(results.begin(), results.end());
            } else if (candidate < results.front()) {
                std::pop_heap(results.begin(), results.end());
                results.back() = candidate;
                std::push_heap(results.begin(), results.end());
            }
        }
    }
    std::sort_heap(results.begin(), results.end());
    for (size_t i = 0; i < want; ++i) {
        out.dists[i] = i < results.size() ? results[i].first : std::numeric_limits<dist_t>::max();
        out.ids[i] = i < results.size() ? results[i].second : kInvalidIdx;
    }
}

size_t IVFIndex::MemoryUsage() const {
    return sizeof(*this) + (raw_vectors_.capacity() + centroids_.capacity()) * sizeof(float) +
           (list_offsets_.capacity() + list_ids_.capacity()) * sizeof(uint32_t);
}

void IVFIndex::SetNProbe(size_t nprobe) noexcept {
    nprobe_.store(std::max(size_t{1}, nprobe), std::memory_order_relaxed);
}

size_t IVFIndex::GetNProbe() const noexcept {
    return nprobe_.load(std::memory_order_relaxed);
}

}  // namespace ann
