#include "hnsw_level.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ann::detail {

double HNSWOpenUnit(uint64_t bits) noexcept {
    // 52 位整数加半格可被 double 精确表示，两端都不会取到 0 或 1。
    return (static_cast<double>(bits >> 12) + 0.5) * 0x1p-52;
}

uint8_t HNSWLevelFromUnit(double unit, double multiplier) {
    if (!(unit > 0.0 && unit < 1.0) || !std::isfinite(multiplier) || multiplier <= 0.0) {
        throw std::invalid_argument("unit must be in (0,1) and multiplier must be finite positive");
    }
    const double level = -std::log(unit) * multiplier;
    constexpr auto max_level = std::numeric_limits<uint8_t>::max();
    if (level >= static_cast<double>(max_level)) {
        return max_level;
    }
    // 已知 level 非负且在 uint8_t 范围内，截断等价于 floor。
    return static_cast<uint8_t>(level);
}

HNSWLevelGenerator::HNSWLevelGenerator(size_t max_neighbors, uint64_t seed)
    : rng_(seed), multiplier_(0.0) {
    if (max_neighbors < 2) {
        throw std::invalid_argument("HNSW upper neighbor capacity must be at least two");
    }
    multiplier_ = 1.0 / std::log(static_cast<double>(max_neighbors));
}

uint8_t HNSWLevelGenerator::Next() {
    return HNSWLevelFromUnit(HNSWOpenUnit(rng_()), multiplier_);
}

}  // namespace ann::detail
