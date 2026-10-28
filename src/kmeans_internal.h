#pragma once
#ifndef ANN_SRC_KMEANS_INTERNAL_H
#define ANN_SRC_KMEANS_INTERNAL_H

#include <cstddef>
#include <span>

namespace ann::detail {

// 仅供训练器和定向测试使用：n>=簇数，至少一个非空簇，缓冲已预分配。
// centers 为簇数×dim；counts 为更新前分配统计，其余缓冲长度为 n。
void ReseedEmptyKMeansClusters(size_t n,
                               size_t dim,
                               const float* data,
                               std::span<float> centers,
                               std::span<const size_t> counts,
                               std::span<double> nearest,
                               std::span<unsigned char> chosen);

}  // namespace ann::detail
#endif  // ANN_SRC_KMEANS_INTERNAL_H
