#include "ann/distance.h"

#include "distance_dispatch.h"

//实现函数
namespace ann {
//欧氏距离平方：Σ (aᵢ − bᵢ)²
dist_t L2SqrScalar(const float* a, const float* b, size_t dim) {
    //定义欧式距离平方
    dist_t l2_sqr = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        l2_sqr += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return l2_sqr;
}

//内积的相反数：− Σ (aᵢ × bᵢ)
dist_t InnerProductScalar(const float* a, const float* b, size_t dim) {
    //定义内积
    dist_t inner_product = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        inner_product += a[i] * b[i];
    }
    return -inner_product;
}

const CpuFeatures& CpuFeatures::Get() {
    // 局部 static 初始化由语言保证线程安全，此后通过 const 引用只读访问。
    static const CpuFeatures features = [] {
        CpuFeatures detected;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        detected.avx2 = __builtin_cpu_supports("avx2") != 0;
        detected.fma = __builtin_cpu_supports("fma") != 0;
#endif
        return detected;
    }();
    return features;
}

DistFn SelectDistFnScalar(Metric metric) {
    if (metric == Metric::L2) {
        return L2SqrScalar;
    }
    // 保持原行为：余弦尚未归一化，未知枚举值也仍回退负内积。
    return InnerProductScalar;
}

namespace detail {

DistFn SelectDistFnForFeatures(Metric metric, const CpuFeatures& features) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (features.avx2 && features.fma) {
        return metric == Metric::L2 ? L2SqrAVX2Four : InnerProductAVX2Four;
    }
#else
    // 当前其他平台只编译标量回退，不把函数名含 AVX2 当作存在 SIMD 后端。
    (void)features;
#endif
    return SelectDistFnScalar(metric);
}

}  // namespace detail

DistFn SelectDistFn(Metric metric) {
    return detail::SelectDistFnForFeatures(metric, CpuFeatures::Get());
}

}  // namespace ann
