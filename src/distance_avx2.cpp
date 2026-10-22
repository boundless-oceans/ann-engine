// 单累加器与四累加器 SIMD：共享水平归约，保留独立入口用于对比。
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

__attribute__((target("avx2,fma"))) dist_t L2SqrAVX2Four(const float* a,
                                                         const float* b,
                                                         size_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const size_t block_end = dim - dim % 32;
    size_t i = 0;
    for (; i < block_end; i += 32) {
        // 四个累加器互不依赖，下一轮的同一累加器才依赖上一轮。
        const __m256 diff0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 diff1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        const __m256 diff2 =
            _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        const __m256 diff3 =
            _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
    }

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    // 不足 32 维时仍可用 SIMD 处理完整的 8 维块，最后最多剩 7 维。
    const size_t vector_end = dim - dim % 8;
    for (; i < vector_end; i += 8) {
        const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    dist_t sum = SumLanes(acc);
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

__attribute__((target("avx2,fma"))) dist_t InnerProductAVX2Four(const float* a,
                                                                const float* b,
                                                                size_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const size_t block_end = dim - dim % 32;
    size_t i = 0;
    for (; i < block_end; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    const size_t vector_end = dim - dim % 8;
    for (; i < vector_end; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
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

dist_t L2SqrAVX2Four(const float* a, const float* b, size_t dim) {
    return L2SqrScalar(a, b, dim);
}

dist_t InnerProductAVX2Four(const float* a, const float* b, size_t dim) {
    return InnerProductScalar(a, b, dim);
}

}  // namespace ann

#endif
