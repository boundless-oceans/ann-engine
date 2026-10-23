// HNSW 简化基线：最近邻连接，后续再接启发式多样性剪枝。
#pragma once
#ifndef ANN_HNSW_H
#define ANN_HNSW_H

#include <memory>

#include "ann/index.h"

namespace ann {

struct HNSWParams {
    size_t M = 16;
    size_t Mmax0 = 32;
    size_t ef_construction = 200;
    size_t ef_search = 64;
    uint64_t seed = 42;
};

class HNSWIndex final : public Index {
public:
    HNSWIndex(size_t dim, Metric metric, HNSWParams params = {});
    ~HNSWIndex() override;
    HNSWIndex(const HNSWIndex&) = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;

    // 单写多读：Add 与任何查询/统计不得并发；批量失败时已完成的前缀保留。
    void Add(size_t n, const float* vecs) override;
    // 当前基线每次创建局部工作区，会分配内存；不修改任何索引成员。
    void Search(const float* query, size_t k, ResultView out) const override;

    size_t Size() const override;
    size_t Dim() const override;
    Metric GetMetric() const override;
    size_t MemoryUsage() const override;
    const char* Name() const override { return "HNSW"; }

    // 零值收敛到 1；运行时 ef 只通过原子变量修改。
    void SetEfSearch(size_t ef) noexcept;
    size_t GetEfSearch() const noexcept;
    // 返回构造参数快照；当前运行时 ef 请调用 GetEfSearch。
    const HNSWParams& Params() const noexcept;

    struct Stats {
        size_t nodes = 0;
        size_t edges_level0 = 0;
        size_t edges_upper = 0;
        int max_level = -1;
        double avg_degree_l0 = 0.0;
    };
    Stats GetStats() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace ann

#endif  // ANN_HNSW_H
