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

void CheckAgainstDouble(const float* a,
                        const float* b,
                        size_t dim,
                        ann::DistFn l2_fn = ann::L2SqrAVX2,
                        ann::DistFn ip_fn = ann::InnerProductAVX2) {
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
    EXPECT_NEAR(static_cast<double>(l2_fn(a, b, dim)), l2, l2_tolerance);
    EXPECT_NEAR(static_cast<double>(ip_fn(a, b, dim)), -ip, ip_tolerance);
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

// 复用同样的 CPU 特性检查，四累加器结果仍与独立 double 参考比较。
class DistanceAVX2Four : public DistanceAVX2 {};

TEST_F(DistanceAVX2Four, HandComputedBlocks) {
    std::array<float, 32> zero{};
    std::array<float, 32> blocks{};
    std::array<float, 32> ones{};
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i] = static_cast<float>(i / 8 + 1);
        ones[i] = 1.0f;
    }
    // 四个分块各有 8 个元素，L2 = 8*(1+4+9+16)，负内积 = -8*(1+2+3+4)。
    EXPECT_NEAR(ann::L2SqrAVX2Four(zero.data(), blocks.data(), 32), 240.0f, 1e-5f);
    EXPECT_NEAR(ann::InnerProductAVX2Four(ones.data(), blocks.data(), 32), -80.0f, 1e-5f);
}

TEST_F(DistanceAVX2Four, ZeroDimensionAndIdenticalVectors) {
    EXPECT_FLOAT_EQ(ann::L2SqrAVX2Four(nullptr, nullptr, 0), 0.0f);
    EXPECT_FLOAT_EQ(ann::InnerProductAVX2Four(nullptr, nullptr, 0), 0.0f);
    const std::vector<float> a(129, 3.25f);
    EXPECT_FLOAT_EQ(ann::L2SqrAVX2Four(a.data(), a.data(), a.size()), 0.0f);
}

TEST_F(DistanceAVX2Four, BlockBoundariesAndRemainders) {
    for (const size_t dim :
         {1u,  7u,  8u,   9u,   15u,  16u,  17u,  23u,  24u,  25u,  31u, 32u, 33u,
          39u, 40u, 41u,  47u,  48u,  49u,  55u,  56u,  57u,  63u,  64u, 65u, 95u,
          96u, 97u, 127u, 128u, 129u, 159u, 160u, 161u, 959u, 960u, 961u}) {
        SCOPED_TRACE(dim);
        // 仅分配实际长度以便 ASan 检测过度加载，最后一维使用不同数值检查遗漏。
        const std::vector<float> a(dim, 1.0f);
        std::vector<float> b(dim, 2.0f);
        b.back() = 3.0f;
        CheckAgainstDouble(a.data(), b.data(), dim, ann::L2SqrAVX2Four, ann::InnerProductAVX2Four);
    }
}

TEST_F(DistanceAVX2Four, UnalignedInputs) {
    alignas(32) std::array<float, 137> a{};
    alignas(32) std::array<float, 137> b{};
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i) / 7.0f;
        b[i] = -static_cast<float>(i) / 11.0f;
    }
    for (size_t offset = 1; offset < 8; ++offset) {
        SCOPED_TRACE(offset);
        CheckAgainstDouble(a.data() + offset,
                           b.data() + offset,
                           129,
                           ann::L2SqrAVX2Four,
                           ann::InnerProductAVX2Four);
    }
}

TEST_F(DistanceAVX2Four, MixedSignsAgainstIndependentReference) {
    std::mt19937 rng(42);
    for (const size_t dim : {31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u, 960u, 961u}) {
        SCOPED_TRACE(dim);
        std::vector<float> a(dim);
        std::vector<float> b(dim);
        for (size_t i = 0; i < dim; ++i) {
            a[i] = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.0f;
            b[i] = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.0f;
        }
        CheckAgainstDouble(a.data(), b.data(), dim, ann::L2SqrAVX2Four, ann::InnerProductAVX2Four);
    }
}

TEST_F(DistanceAVX2Four, InnerProductCancellation) {
    std::array<float, 33> a{};
    std::array<float, 33> b{};
    for (size_t i = 0; i < 32; ++i) {
        a[i] = 10000.0f;
        b[i] = i % 2 == 0 ? 10000.0f : -10000.0f;
    }
    a[32] = 1.0f;
    b[32] = 0.25f;
    CheckAgainstDouble(a.data(), b.data(), a.size(), ann::L2SqrAVX2Four, ann::InnerProductAVX2Four);
    EXPECT_NEAR(ann::InnerProductAVX2Four(a.data(), b.data(), a.size()), -0.25f, 1e-5f);
}

}  // namespace
