#include "ann/quantizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ann {
namespace {

constexpr size_t kLevels = 256;
constexpr double kMaxCode = 255.0;

void CheckFinite(const float* values, size_t count) {
    if (values == nullptr) {
        throw std::invalid_argument("SQ8 requires a non-null input buffer");
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            throw std::invalid_argument("SQ8 requires finite coordinates");
        }
    }
}

float Reconstruct(float low, float high, uint8_t code) noexcept {
    // lerp 保持端点和范围，宽范围不先在 float 中求 high-low。
    return static_cast<float>(std::lerp(
        static_cast<double>(low), static_cast<double>(high), static_cast<double>(code) / kMaxCode));
}

}  // namespace

ScalarQuantizer::ScalarQuantizer(size_t dim) : dim_(dim) {
    if (dim == 0) {
        throw std::invalid_argument("SQ8 dimension must be positive");
    }
    if (dim > std::numeric_limits<size_t>::max() / kLevels / sizeof(float)) {
        throw std::length_error("SQ8 LUT storage size overflow");
    }
}

void ScalarQuantizer::RequireTrained() const {
    if (!trained_) {
        throw std::logic_error("SQ8 operation requires Train");
    }
}

void ScalarQuantizer::Train(size_t n, const float* vecs) {
    if (n == 0 || vecs == nullptr) {
        throw std::invalid_argument("SQ8 training requires nonempty data");
    }
    if (n > std::numeric_limits<size_t>::max() / dim_ / sizeof(float)) {
        throw std::length_error("SQ8 training matrix size overflow");
    }
    std::vector<float> mins(dim_, std::numeric_limits<float>::max());
    std::vector<float> maxs(dim_, std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim_; ++d) {
            const float value = vecs[i * dim_ + d];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("SQ8 requires finite training coordinates");
            }
            mins[d] = std::min(mins[d], value);
            maxs[d] = std::max(maxs[d], value);
        }
    }
    mins_.swap(mins);
    maxs_.swap(maxs);
    trained_ = true;
}

void ScalarQuantizer::Encode(const float* vec, uint8_t* code) const {
    RequireTrained();
    if (code == nullptr) {
        throw std::invalid_argument("SQ8 requires a non-null code buffer");
    }
    CheckFinite(vec, dim_);
    for (size_t d = 0; d < dim_; ++d) {
        if (mins_[d] >= maxs_[d] || vec[d] <= mins_[d]) {
            code[d] = 0;
        } else if (vec[d] >= maxs_[d]) {
            code[d] = 255;
        } else {
            const double numerator = static_cast<double>(vec[d]) - static_cast<double>(mins_[d]);
            const double range = static_cast<double>(maxs_[d]) - static_cast<double>(mins_[d]);
            const double mapped = std::clamp(numerator / range * kMaxCode, 0.0, kMaxCode);
            code[d] = static_cast<uint8_t>(std::floor(mapped + 0.5));
        }
    }
}

void ScalarQuantizer::Decode(const uint8_t* code, float* vec) const {
    RequireTrained();
    if (code == nullptr || vec == nullptr) {
        throw std::invalid_argument("SQ8 requires non-null decode buffers");
    }
    for (size_t d = 0; d < dim_; ++d) {
        vec[d] = Reconstruct(mins_[d], maxs_[d], code[d]);
    }
}

void ScalarQuantizer::BuildLUT(const float* query, float* lut) const {
    RequireTrained();
    if (lut == nullptr) {
        throw std::invalid_argument("SQ8 requires a non-null LUT buffer");
    }
    CheckFinite(query, dim_);
    // 平方误差在区间端点取最大值。先检查全部维度，失败时不写半张表。
    for (size_t d = 0; d < dim_; ++d) {
        const double low_diff = static_cast<double>(query[d]) - static_cast<double>(mins_[d]);
        const double high_diff = static_cast<double>(query[d]) - static_cast<double>(maxs_[d]);
        if (std::max(low_diff * low_diff, high_diff * high_diff) >
            static_cast<double>(std::numeric_limits<float>::max())) {
            throw std::overflow_error("SQ8 squared distance exceeds float LUT range");
        }
    }
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t c = 0; c < kLevels; ++c) {
            const double diff =
                static_cast<double>(query[d]) -
                static_cast<double>(Reconstruct(mins_[d], maxs_[d], static_cast<uint8_t>(c)));
            lut[d * kLevels + c] = static_cast<float>(diff * diff);
        }
    }
}

float ScalarQuantizer::DistanceFromLUT(const float* lut, const uint8_t* code) const {
    RequireTrained();
    if (lut == nullptr || code == nullptr) {
        throw std::invalid_argument("SQ8 requires non-null distance buffers");
    }
    double sum = 0;
    for (size_t d = 0; d < dim_; ++d) {
        const float value = lut[d * kLevels + code[d]];
        if (!std::isfinite(value) || value < 0) {
            throw std::invalid_argument("SQ8 LUT entries must be finite nonnegative distances");
        }
        sum += static_cast<double>(value);
    }
    if (sum > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::overflow_error("SQ8 total distance exceeds float range");
    }
    return static_cast<float>(sum);
}

size_t ScalarQuantizer::MemoryUsage() const {
    return sizeof(*this) + (mins_.capacity() + maxs_.capacity()) * sizeof(float);
}

}  // namespace ann
