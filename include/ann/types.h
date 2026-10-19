// 基础类型与常量。库内所有代码统一使用 ann 命名空间。
#pragma once

// 与包含卫士并用：前者是快路径（编译器直接跳过重复包含），后者在同文件经不同路径被包含时兜底
#ifndef ANN_TYPES_H
#define ANN_TYPES_H

#include <cstddef>
#include <cstdint>
#include <limits>

// 编译期确认 C++20 已启用。
// 若构建脚本漏传标准旗标，这里会立刻失败，而不是等到某个特性用不了才发现。
static_assert(__cplusplus >= 202002L, "ann-engine requires C++20");

namespace ann {

// 向量 ID 类型。
// 约定：内部引用一律使用下标，不使用指针 —— 既有利于 cache，
// 也让持久化与未来的分片无需处理指针重定位。
using idx_t = uint32_t;

// 距离值类型
using dist_t = float;

// 无效下标哨兵
inline constexpr idx_t kInvalidIdx = std::numeric_limits<idx_t>::max();

// SIMD 对齐字节数。AVX2 需要 32 字节。
inline constexpr size_t kAlignment = 32;

// 距离度量
enum class Metric : uint8_t {
    L2 = 0,            // 欧氏距离平方
    InnerProduct = 1,  // 内积
    Cosine = 2,        // 余弦（需先归一化，暂未实现）
};

// 向量的只读视图，不拥有数据
struct VectorView {
    const float* data = nullptr;
    size_t dim = 0;
};

// 查询结果的输出参数。
// 由调用方预分配，索引只负责填充 —— 返回 std::vector 会在热路径引入分配，
// 使性能测量结果失真。
struct ResultView {
    dist_t* dists = nullptr;  // 容量 >= k
    idx_t* ids = nullptr;     // 容量 >= k
    size_t k = 0;             // 期望返回的条数
};

}  // namespace ann

#endif  // ANN_TYPES_H
