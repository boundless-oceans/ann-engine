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

void Residual(const float* vector, const float* center, size_t dim, float* out) {
    for (size_t d = 0; d < dim; ++d) {
        const double value = static_cast<double>(vector[d]) - static_cast<double>(center[d]);
        if (std::abs(value) > static_cast<double>(std::numeric_limits<float>::max())) {
            throw std::overflow_error("IVF residual exceeds float range");
        }
        out[d] = static_cast<float>(value);
    }
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
        throw std::invalid_argument("IVF currently supports L2 only");
    }
    if (quantizer) {
        if (quantizer->Dim() != dim) {
            throw std::invalid_argument("IVF quantizer dimension must match the index");
        }
        // 形状统一走基类契约，类型检查仅限定本阶段支持的内置实现/报告名称。
        if (dynamic_cast<const ScalarQuantizer*>(quantizer.get()) != nullptr) {
            quantizer_kind_ = QuantizerKind::SQ8;
        } else if (dynamic_cast<const ProductQuantizer*>(quantizer.get()) != nullptr) {
            quantizer_kind_ = QuantizerKind::PQ;
        } else {
            throw std::invalid_argument("IVF currently supports built-in SQ8 or PQ quantizers");
        }
        code_size_ = quantizer->CodeSize();
    }
    if (dim > std::numeric_limits<size_t>::max() / sizeof(float) / params.nlist) {
        throw std::length_error("IVF centroid storage size overflow");
    }
    quantizer_ = std::move(quantizer);
}

size_t IVFIndex::Size() const {
    return n_;
}

const char* IVFIndex::Name() const {
    switch (quantizer_kind_) {
        case QuantizerKind::None:
            return "IVFFlat";
        case QuantizerKind::SQ8:
            return "IVFSQ8";
        case QuantizerKind::PQ:
            return "IVFPQ";
    }
    return "IVF";
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
    n_ += n;
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
    std::vector<uint8_t> codes;
    if (quantizer_) {
        // 共享一个全局残差量化器；整批残差和原向量不进入构建后的持久存储。
        std::vector<float> residuals(raw_vectors_.size());
        for (size_t id = 0; id < n; ++id) {
            Residual(raw_vectors_.data() + id * dim_,
                     centers.data() + size_t{assignments[id]} * dim_,
                     dim_,
                     residuals.data() + id * dim_);
        }
        if (n > codes.max_size() / code_size_) {
            throw std::length_error("IVF encoded storage size overflow");
        }
        codes.resize(n * code_size_);
        // 失败不发布索引结构；私有量化器可能已完成训练，重试时会重新 Train。
        quantizer_->Train(n, residuals.data());
        for (size_t id = 0; id < n; ++id) {
            quantizer_->Encode(residuals.data() + id * dim_, codes.data() + id * code_size_);
        }
    }
    centroids_.swap(centers);
    list_offsets_.swap(lists.offsets);
    list_ids_.swap(lists.ids);
    codes_.swap(codes);
    if (quantizer_) {
        std::vector<float>().swap(raw_vectors_);
    }
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
    std::vector<float> residual_query, lut;
    if (quantizer_) {
        residual_query.resize(dim_);
        lut.resize(quantizer_->LutSize());
    }
    for (size_t p = 0; p < probes; ++p) {
        const size_t c = probes == params_.nlist ? p : clusters[p].second;
        if (list_offsets_[c] == list_offsets_[c + 1]) {
            continue;
        }
        if (quantizer_) {
            Residual(query, centroids_.data() + c * dim_, dim_, residual_query.data());
            quantizer_->BuildLUT(residual_query.data(), lut.data());
        }
        for (size_t pos = list_offsets_[c]; pos < list_offsets_[c + 1]; ++pos) {
            const idx_t id = list_ids_[pos];
            const dist_t distance =
                quantizer_ ? quantizer_->DistanceFromLUT(lut.data(),
                                                         codes_.data() + size_t{id} * code_size_)
                           : CheckedDistance(
                                 dist_fn_, query, raw_vectors_.data() + size_t{id} * dim_, dim_);
            const Candidate candidate{distance, id};
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
           (list_offsets_.capacity() + list_ids_.capacity()) * sizeof(uint32_t) +
           codes_.capacity() * sizeof(uint8_t) + (quantizer_ ? quantizer_->MemoryUsage() : 0);
}

void IVFIndex::SetNProbe(size_t nprobe) noexcept {
    nprobe_.store(std::max(size_t{1}, nprobe), std::memory_order_relaxed);
}

size_t IVFIndex::GetNProbe() const noexcept {
    return nprobe_.load(std::memory_order_relaxed);
}

}  // namespace ann
