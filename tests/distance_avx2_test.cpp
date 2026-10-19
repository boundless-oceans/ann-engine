// 用独立 double 参考复核 SIMD，避免仅比较两份 float 实现。
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "ann/distance.h"

namespace {

class DistanceAVX2 : public testing::Test {
protected:
    void SetUp() override {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
            GTEST_SKIP() << "AVX2 and FMA are required";
        }
#else
        GTEST_SKIP() << "AVX2 implementation requires x86 GCC/Clang";
#endif
    }
};

void CheckAgainstDouble(const float* a, const float* b, size_t dim) {
    double l2 = 0.0;
    double ip = 0.0;
    double abs_products = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double x = static_cast<double>(a[i]);
        const double y = static_cast<double>(b[i]);
        l2 += (x - y) * (x - y);
        ip += x * y;
        abs_products += std::abs(x * y);
    }
    // 内积可能相消到接近零，因此按绝对乘积和设误差尺度，不只看最终结果。
    const double l2_tolerance = 2e-5 * std::max(1.0, l2);
    const double ip_tolerance = 2e-5 * std::max(1.0, abs_products);
    EXPECT_NEAR(static_cast<double>(ann::L2SqrAVX2(a, b, dim)), l2, l2_tolerance);
    EXPECT_NEAR(static_cast<double>(ann::InnerProductAVX2(a, b, dim)), -ip, ip_tolerance);
    EXPECT_NEAR(static_cast<double>(ann::L2SqrScalar(a, b, dim)), l2, l2_tolerance);
    EXPECT_NEAR(static_cast<double>(ann::InnerProductScalar(a, b, dim)), -ip, ip_tolerance);
}

TEST_F(DistanceAVX2, L2HandComputed) {
    // 八维全部为 (3,4) 重复：4 * (9 + 16) = 100，覆盖向量主体和水平求和。
    const float a[8]{};
    const float b[8] = {3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f};
    EXPECT_NEAR(ann::L2SqrAVX2(a, b, 8), 100.0f, 1e-5f);
    EXPECT_NEAR(ann::L2SqrAVX2(b, a, 8), 100.0f, 1e-5f);
}

TEST_F(DistanceAVX2, NegativeInnerProductHandComputed) {
    const float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float b[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_NEAR(ann::InnerProductAVX2(a, b, 8), -36.0f, 1e-5f);
}

TEST_F(DistanceAVX2, ZeroDimensionDoesNotReadInput) {
    EXPECT_FLOAT_EQ(ann::L2SqrAVX2(nullptr, nullptr, 0), 0.0f);
    EXPECT_FLOAT_EQ(ann::InnerProductAVX2(nullptr, nullptr, 0), 0.0f);
}

TEST_F(DistanceAVX2, IdenticalVectorsHaveZeroL2) {
    const std::vector<float> a(129, 3.25f);
    EXPECT_FLOAT_EQ(ann::L2SqrAVX2(a.data(), a.data(), a.size()), 0.0f);
}

TEST_F(DistanceAVX2, VectorBoundariesAndTails) {
    for (const size_t dim :
         {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 127u, 128u, 129u, 960u}) {
        SCOPED_TRACE(dim);
        // 按实际维度分配，配合 ASan 检查尾部是否越界读取。
        const std::vector<float> a(dim, 1.0f);
        const std::vector<float> b(dim, 2.0f);
        EXPECT_NEAR(ann::L2SqrAVX2(a.data(), b.data(), dim), static_cast<float>(dim), 1e-5f);
        EXPECT_NEAR(
            ann::InnerProductAVX2(a.data(), b.data(), dim), -2.0f * static_cast<float>(dim), 1e-5f);
    }
}

TEST_F(DistanceAVX2, UnalignedInputs) {
    alignas(32) std::array<float, 137> a{};
    alignas(32) std::array<float, 137> b{};
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i) / 7.0f;
        b[i] = -static_cast<float>(i) / 11.0f;
    }
    for (size_t offset = 1; offset < 8; ++offset) {
        SCOPED_TRACE(offset);
        CheckAgainstDouble(a.data() + offset, b.data() + offset, 129);
    }
}

TEST_F(DistanceAVX2, MixedSignsAgainstIndependentReference) {
    std::mt19937 rng(42);
    for (const size_t dim : {7u, 8u, 17u, 127u, 128u, 129u, 960u}) {
        SCOPED_TRACE(dim);
        std::vector<float> a(dim);
        std::vector<float> b(dim);
        for (size_t i = 0; i < dim; ++i) {
            a[i] = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.0f;
            b[i] = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.0f;
        }
        CheckAgainstDouble(a.data(), b.data(), dim);
    }
}

TEST_F(DistanceAVX2, InnerProductCancellation) {
    const float a[9] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1};
    const float b[9] = {10000, -10000, 10000, -10000, 10000, -10000, 10000, -10000, 0.25f};
    CheckAgainstDouble(a, b, 9);
    EXPECT_NEAR(ann::InnerProductAVX2(a, b, 9), -0.25f, 1e-5f);
}

}  // namespace
