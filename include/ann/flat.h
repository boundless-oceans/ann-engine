// 暴力检索索引。穷举精确距离，严格 ID Recall 仍受等距点选择影响。
// 单次查询扫描的距离计算复杂度 O(n × dim)。
//
// 两个用途：
//   1. 性能基准 —— 后续所有优化的对照点
//   2. ground truth 生成器 —— 近似索引无法用「结果相等」验证正确性，
//      只能与这里的精确结果比对召回率
#pragma once

// 与包含卫士并用：前者是快路径（编译器直接跳过重复包含），后者在同文件经不同路径被包含时兜底
#ifndef ANN_FLAT_H
#define ANN_FLAT_H

#include "ann/distance.h"
#include "ann/index.h"
#include "ann/memory_pool.h"

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
    const float* Data() const noexcept { return data_.Data(); }

private:
    size_t dim_;
    Metric metric_;
    DistFn dist_fn_;

    // 全部向量连续存放（n × dim）。
    // 数组起点按 32 字节对齐；无行填充，非八倍数维度的后续行仍可能不对齐。
    AlignedVector<float> data_;
    size_t n_ = 0;
};

}  // namespace ann

#endif  // ANN_FLAT_H
