// 距离函数的声明。实现见 src/distance.cpp。
//
// 排序约定：所有距离函数遵循「值越小表示越相似」。
//   - L2 平方距离天然满足
//   - 内积方向相反，因此内积实现返回其相反数（-<a,b>）
// 索引实现只需按升序取前 k 个，无需关心具体度量。
#pragma once

#include <cstddef>

#include "ann/types.h"

namespace ann {

// 距离函数指针类型。标量与 AVX2 实现共用同一签名。
using DistFn = dist_t (*)(const float* a, const float* b, size_t dim);

// ---- 标量参考实现 ----
// 同时也是 AVX2 版本的正确性基准：两者结果必须一致。
dist_t L2SqrScalar(const float* a, const float* b, size_t dim);
dist_t InnerProductScalar(const float* a, const float* b, size_t dim);

// ---- AVX2 实现（后续加入）----
// 新增后需在 SelectDistFn 中按 CPU 特性分派，并要求操作数 32 字节对齐、
// dim 为 8 的倍数，否则退回标量路径。
// dist_t L2SqrAVX2(const float* a, const float* b, size_t dim);
// dist_t InnerProductAVX2(const float* a, const float* b, size_t dim);

// 按度量选择距离函数。
// 返回的函数已按上述排序约定归一化：内积返回负值。
// Metric::Cosine 尚未实现，当前会回退到内积。
DistFn SelectDistFn(Metric metric);

}  // namespace ann
