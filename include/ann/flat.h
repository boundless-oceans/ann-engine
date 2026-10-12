// 暴力检索索引。精确，Recall 恒为 1.0，单次查询复杂度 O(n × dim)。
//
// 两个用途：
//   1. 性能基准 —— 后续所有优化的对照点
//   2. ground truth 生成器 —— 近似索引无法用「结果相等」验证正确性，
//      只能与这里的精确结果比对召回率
#pragma once

#include <vector>

#include "ann/distance.h"
#include "ann/index.h"

namespace ann {

class FlatIndex final : public Index {
public:
    FlatIndex(size_t dim, Metric metric);

    void Add(size_t n, const float* vecs) override;
    void Search(const float* query, size_t k, ResultView out) const override;

    size_t Size() const override;
    size_t Dim() const override;
    Metric GetMetric() const override;
    size_t MemoryUsage() const override;
    const char* Name() const override { return "Flat"; }

    // 只读访问原始向量数据。bench 用它计算查询与库向量的配对情况。
    const float* Data() const noexcept { return data_.data(); }

private:
    size_t dim_;
    Metric metric_;
    DistFn dist_fn_;

    // 全部向量连续存放（n × dim）。
    // 注意：当前使用 std::vector，未做 SIMD 对齐。
    // 引入 AVX2 后需替换为 32 字节对齐的容器，并改用对齐加载指令。
    std::vector<float> data_;
    size_t n_ = 0;
};

}  // namespace ann
