// 第一版 SIMD：每轮处理 8 维，保留单个累加器作为后续四累加器的对照。
#include "ann/distance.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

namespace ann {
namespace {

// 指令集仅作用于此函数，避免给整个核心库添加 -mavx2 / -mfma。
__attribute__((target("avx2,fma"))) float SumLanes(__m256 value) {
    const __m128 lo = _mm256_castps256_ps128(value);
    const __m128 hi = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

}  // namespace

__attribute__((target("avx2,fma"))) dist_t L2SqrAVX2(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    const size_t vector_end = dim - dim % 8;
    size_t i = 0;
    for (; i < vector_end; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    dist_t sum = SumLanes(acc);
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

__attribute__((target("avx2,fma"))) dist_t InnerProductAVX2(const float* a,
                                                            const float* b,
                                                            size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    const size_t vector_end = dim - dim % 8;
    size_t i = 0;
    for (; i < vector_end; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    dist_t sum = SumLanes(acc);
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return -sum;
}

}  // namespace ann

#else

namespace ann {

dist_t L2SqrAVX2(const float* a, const float* b, size_t dim) {
    return L2SqrScalar(a, b, dim);
}

dist_t InnerProductAVX2(const float* a, const float* b, size_t dim) {
    return InnerProductScalar(a, b, dim);
}

}  // namespace ann

#endif
