#pragma once
#ifndef ANN_KMEANS_H
#define ANN_KMEANS_H

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ann {

// L2 k-means：供 IVF 粗聚类和 PQ 子空间训练复用。
class KMeans {
public:
    KMeans(size_t dim, size_t k, size_t iters = 25, uint64_t seed = 42);

    // 输入为连续 n×dim，输出容量至少 k×dim；要求 n>=k、输入有限。
    // 完整成功后才写输出和簇统计；输出可以与输入重叠。
    void Train(size_t n, const float* data, float* out_centroids);

    // 等距选择较小簇 ID。n=0 允许空输入/输出，且不读取质心。
    void Assign(size_t n, const float* data, const float* centroids, uint32_t* out_assign) const;

    // 最近一次成功 Train 的最终质心分配统计；训练前为空。
    // 重复样本可能导致最终空簇，重置质心不保证所有簇都有成员。
    const std::vector<size_t>& ClusterSizes() const noexcept { return cluster_sizes_; }

private:
    size_t dim_, k_, iters_;
    std::mt19937_64 rng_;
    std::vector<size_t> cluster_sizes_;
};

}  // namespace ann
#endif  // ANN_KMEANS_H
