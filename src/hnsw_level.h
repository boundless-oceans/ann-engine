// HNSW 内部层数生成器，只在构建时推进随机状态。
#pragma once
#ifndef ANN_SRC_HNSW_LEVEL_H
#define ANN_SRC_HNSW_LEVEL_H

#include <cstddef>
#include <cstdint>
#include <random>

namespace ann::detail {

// 用高 52 位映射到开区间 (0,1) 的中点网格；每次消耗一个随机整数。
double HNSWOpenUnit(uint64_t bits) noexcept;

// 纯转换供边界测试复用；要求 0<unit<1，multiplier 有限且为正。
// 按 uint8_t 图层元数据范围饱和，不允许浮点到整数转换越界。
uint8_t HNSWLevelFromUnit(double unit, double multiplier);

class HNSWLevelGenerator {
public:
    // M 是上层邻居容量，必须至少为 2；固定种子便于复现构建。
    explicit HNSWLevelGenerator(size_t max_neighbors, uint64_t seed = 42);
    uint8_t Next();

private:
    std::mt19937_64 rng_;
    double multiplier_;
};

}  // namespace ann::detail

#endif  // ANN_SRC_HNSW_LEVEL_H
