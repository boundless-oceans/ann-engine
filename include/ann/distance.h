// 距离函数的声明。实现见 src/distance.cpp。
//
// 排序约定：所有距离函数遵循「值越小表示越相似」。
//   - L2 平方距离天然满足
//   - 内积方向相反，因此内积实现返回其相反数（-<a,b>）
// 索引实现只需按升序取前 k 个，无需关心具体度量。
#pragma once

// 与包含卫士并用：前者是快路径（编译器直接跳过重复包含），后者在同文件经不同路径被包含时兜底
#ifndef ANN_DISTANCE_H
#define ANN_DISTANCE_H

#include <cstddef>

#include "ann/types.h"

namespace ann {

// 距离函数指针类型。标量与 AVX2 实现共用同一签名。
using DistFn = dist_t (*)(const float* a, const float* b, size_t dim);

// ---- 标量参考实现 ----
// 同时也是 AVX2 版本的正确性参考：浮点归约顺序不同，结果按误差容限比较。
dist_t L2SqrScalar(const float* a, const float* b, size_t dim);
dist_t InnerProductScalar(const float* a, const float* b, size_t dim);

// ---- AVX2 + FMA 单累加器实现 ----
// x86 GCC/Clang 下直接调用前必须确认 CPU 支持 AVX2 与 FMA。
// 输入只需满足 float 自身对齐，使用 loadu；任意维度均支持，尾部走标量循环。
// dim=0 不读取输入；其他情况下两侧至少有 dim 个可读 float。
// 其他平台/编译器暂回退标量，不代表其已有 SIMD 实现。
// 直接调用这些入口仍需调用者检查 CPU；SelectDistFn 提供安全自动选择。
dist_t L2SqrAVX2(const float* a, const float* b, size_t dim);
dist_t InnerProductAVX2(const float* a, const float* b, size_t dim);

// ---- AVX2 + FMA 四累加器实现 ----
// 与单累加器契约一致；每轮处理 32 维，用四条独立依赖链隐藏 FMA 延迟。
// 保留独立入口供同轮对比；可用时由 SelectDistFn 自动选择。
dist_t L2SqrAVX2Four(const float* a, const float* b, size_t dim);
dist_t InnerProductAVX2Four(const float* a, const float* b, size_t dim);

// 仅描述本项目当前使用的两项特性；Get 返回首次探测后不再修改的缓存。
struct CpuFeatures {
    bool avx2 = false;
    bool fma = false;
    static const CpuFeatures& Get();
};

// 按度量和本机 CPU 选择距离函数。AVX2/FMA 均可用时选四累加器，否则标量。
// FlatIndex 在构造时保存返回指针，Search 不再次检测 CPU。
// 返回的函数已按上述排序约定归一化：内积返回负值。
// Metric::Cosine 尚未实现，当前会回退到内积。
DistFn SelectDistFn(Metric metric);

// 强制选择标量函数，供正确性参考及独立基线调用，不修改全局派发状态。
DistFn SelectDistFnScalar(Metric metric);

}  // namespace ann

#endif  // ANN_DISTANCE_H
