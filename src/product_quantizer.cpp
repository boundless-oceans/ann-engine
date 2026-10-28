#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "ann/kmeans.h"
#include "ann/quantizer.h"

namespace ann {
namespace {

constexpr size_t kTrainingIterations = 25;

void CheckPQInput(const float* input, size_t count) {
    if (input == nullptr) {
        throw std::invalid_argument("PQ requires a non-null input buffer");
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(input[i])) {
            throw std::invalid_argument("PQ requires finite coordinates");
        }
    }
}

double SubspaceDistance(const float* a, const float* b, size_t dim) {
    double distance = 0;
    for (size_t d = 0; d < dim; ++d) {
        const double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        distance += diff * diff;
    }
    return distance;
}

}  // namespace

ProductQuantizer::ProductQuantizer(size_t dim, size_t m, size_t ksub, uint64_t seed)
    : dim_(dim), m_(m), dsub_(0), ksub_(ksub), seed_(seed) {
    if (dim == 0 || m == 0 || dim % m != 0 || ksub == 0 || ksub > 256) {
        throw std::invalid_argument(
            "PQ requires positive dim/m, dim divisible by m, and 1<=ksub<=256");
    }
    if (dim > std::numeric_limits<size_t>::max() / sizeof(float) / ksub ||
        m > std::numeric_limits<size_t>::max() / sizeof(float) / ksub) {
        throw std::length_error("PQ codebook or LUT byte size overflow");
    }
    dsub_ = dim / m;
}

void ProductQuantizer::RequireTrained() const {
    if (!trained_) {
        throw std::logic_error("PQ operation requires Train");
    }
}

void ProductQuantizer::CheckCodes(const uint8_t* code) const {
    if (code == nullptr) {
        throw std::invalid_argument("PQ requires a non-null code buffer");
    }
    for (size_t s = 0; s < m_; ++s) {
        if (code[s] >= ksub_) {
            throw std::invalid_argument("PQ code exceeds subspace codebook size");
        }
    }
}

void ProductQuantizer::Train(size_t n, const float* vecs) {
    if (n < ksub_) {
        throw std::invalid_argument("PQ training requires at least ksub vectors");
    }
    if (n > std::numeric_limits<size_t>::max() / sizeof(float) / dim_) {
        throw std::length_error("PQ training matrix byte size overflow");
    }
    CheckPQInput(vecs, n * dim_);
    std::vector<float> codebooks(ksub_ * dim_);
    // 复用一段连续训练矩阵，而非同时拷贝m个子空间。
    std::vector<float> subspace(n * dsub_);
    for (size_t s = 0; s < m_; ++s) {
        for (size_t i = 0; i < n; ++i) {
            std::copy_n(vecs + i * dim_ + s * dsub_, dsub_, subspace.data() + i * dsub_);
        }
        // 每次Train从配置种子开始；无符号加法按模2^64定义，不依赖共享RNG。
        KMeans trainer(dsub_, ksub_, kTrainingIterations, seed_ + static_cast<uint64_t>(s));
        trainer.Train(n, subspace.data(), codebooks.data() + s * ksub_ * dsub_);
    }
    codebooks_.swap(codebooks);
    trained_ = true;
}

void ProductQuantizer::Encode(const float* vec, uint8_t* code) const {
    RequireTrained();
    if (code == nullptr) {
        throw std::invalid_argument("PQ requires a non-null code buffer");
    }
    CheckPQInput(vec, dim_);
    for (size_t s = 0; s < m_; ++s) {
        const float* centers = codebooks_.data() + s * ksub_ * dsub_;
        size_t best = 0;
        double nearest = SubspaceDistance(vec + s * dsub_, centers, dsub_);
        for (size_t c = 1; c < ksub_; ++c) {
            const double distance = SubspaceDistance(vec + s * dsub_, centers + c * dsub_, dsub_);
            if (distance < nearest) {
                nearest = distance;
                best = c;
            }
        }
        code[s] = static_cast<uint8_t>(best);
    }
}

void ProductQuantizer::Decode(const uint8_t* code, float* vec) const {
    RequireTrained();
    CheckCodes(code);
    if (vec == nullptr) {
        throw std::invalid_argument("PQ requires a non-null decode buffer");
    }
    for (size_t s = 0; s < m_; ++s) {
        std::copy_n(codebooks_.data() + (s * ksub_ + code[s]) * dsub_, dsub_, vec + s * dsub_);
    }
}

void ProductQuantizer::BuildLUT(const float* query, float* lut) const {
    RequireTrained();
    if (lut == nullptr) {
        throw std::invalid_argument("PQ requires a non-null LUT buffer");
    }
    CheckPQInput(query, dim_);
    // 两遍计算避免局部表项溢出时留下半张表；不用隐藏的临时分配。
    for (size_t s = 0; s < m_; ++s) {
        for (size_t c = 0; c < ksub_; ++c) {
            const auto distance = SubspaceDistance(
                query + s * dsub_, codebooks_.data() + (s * ksub_ + c) * dsub_, dsub_);
            if (distance > static_cast<double>(std::numeric_limits<float>::max())) {
                throw std::overflow_error("PQ squared distance exceeds float LUT range");
            }
        }
    }
    for (size_t s = 0; s < m_; ++s) {
        for (size_t c = 0; c < ksub_; ++c) {
            lut[s * ksub_ + c] = static_cast<float>(SubspaceDistance(
                query + s * dsub_, codebooks_.data() + (s * ksub_ + c) * dsub_, dsub_));
        }
    }
}

float ProductQuantizer::DistanceFromLUT(const float* lut, const uint8_t* code) const {
    RequireTrained();
    CheckCodes(code);
    if (lut == nullptr) {
        throw std::invalid_argument("PQ requires a non-null LUT buffer");
    }
    double sum = 0;
    for (size_t s = 0; s < m_; ++s) {
        const float value = lut[s * ksub_ + code[s]];
        if (!std::isfinite(value) || value < 0) {
            throw std::invalid_argument("PQ LUT entries must be finite nonnegative distances");
        }
        sum += static_cast<double>(value);
    }
    if (sum > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::overflow_error("PQ total distance exceeds float range");
    }
    return static_cast<float>(sum);
}

size_t ProductQuantizer::MemoryUsage() const {
    return sizeof(*this) + codebooks_.capacity() * sizeof(float);
}

}  // namespace ann
