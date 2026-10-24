#pragma once
#ifndef ANN_IVF_H
#define ANN_IVF_H

#include <atomic>
#include <memory>
#include <vector>

#include "ann/distance.h"
#include "ann/index.h"
#include "ann/quantizer.h"

namespace ann {

struct IVFParams {
    size_t nlist = 1024;
    size_t nprobe = 16;
    size_t kmeans_iters = 25;
    uint64_t seed = 42;
};

// 当前支持未量化 L2；先 Add 后 Build，成功构建后不再允许 Add。
class IVFIndex final : public Index {
public:
    IVFIndex(size_t dim,
             Metric metric,
             IVFParams params = {},
             std::unique_ptr<Quantizer> quantizer = nullptr);

    void Add(size_t n, const float* vecs) override;
    // 要求样本数至少 nlist；成功后重复调用无操作，失败可继续 Add/重试。
    void Build() override;
    // 查询前必须 Build；使用局部工作区，有内存分配，不修改成员。
    void Search(const float* query, size_t k, ResultView out) const override;

    size_t Size() const override;
    size_t Dim() const override { return dim_; }
    Metric GetMetric() const override { return metric_; }
    // 统计对象及成员数组申请容量，不含查询/构建工作区和分配器内部开销。
    size_t MemoryUsage() const override;
    const char* Name() const override { return "IVFFlat"; }

    // 零值收敛到 1，超出 nlist 的值在查询时截断；每次查询读一次快照。
    void SetNProbe(size_t nprobe) noexcept;
    size_t GetNProbe() const noexcept;

private:
    size_t dim_;
    Metric metric_;
    IVFParams params_;
    DistFn dist_fn_;
    std::atomic<size_t> nprobe_;
    bool built_ = false;
    std::vector<float> centroids_;
    std::vector<uint32_t> list_offsets_;
    std::vector<uint32_t> list_ids_;
    std::vector<float> raw_vectors_;
};

}  // namespace ann
#endif  // ANN_IVF_H
