#include "distance_dispatch.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "ann/distance.h"
#include "ann/flat.h"

namespace {

void ExpectScalarSelection(ann::CpuFeatures features) {
    EXPECT_EQ(ann::detail::SelectDistFnForFeatures(ann::Metric::L2, features), ann::L2SqrScalar);
    EXPECT_EQ(ann::detail::SelectDistFnForFeatures(ann::Metric::InnerProduct, features),
              ann::InnerProductScalar);
    EXPECT_EQ(ann::detail::SelectDistFnForFeatures(ann::Metric::Cosine, features),
              ann::InnerProductScalar);
}

TEST(DistanceDispatch, CpuFeaturesAreCachedAndMatchHost) {
    const auto& first = ann::CpuFeatures::Get();
    const auto& second = ann::CpuFeatures::Get();
    EXPECT_EQ(&first, &second);
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    EXPECT_EQ(first.avx2, __builtin_cpu_supports("avx2") != 0);
    EXPECT_EQ(first.fma, __builtin_cpu_supports("fma") != 0);
#else
    EXPECT_FALSE(first.avx2);
    EXPECT_FALSE(first.fma);
#endif
}

TEST(DistanceDispatch, ScalarSelectorPreservesMetricConventions) {
    EXPECT_EQ(ann::SelectDistFnScalar(ann::Metric::L2), ann::L2SqrScalar);
    EXPECT_EQ(ann::SelectDistFnScalar(ann::Metric::InnerProduct), ann::InnerProductScalar);
    EXPECT_EQ(ann::SelectDistFnScalar(ann::Metric::Cosine), ann::InnerProductScalar);
    EXPECT_EQ(ann::SelectDistFnScalar(static_cast<ann::Metric>(255)), ann::InnerProductScalar);
}

TEST(DistanceDispatch, NoFeaturesSelectScalar) {
    ExpectScalarSelection({false, false});
}

TEST(DistanceDispatch, Avx2WithoutFmaSelectsScalar) {
    ExpectScalarSelection({true, false});
}

TEST(DistanceDispatch, FmaWithoutAvx2SelectsScalar) {
    ExpectScalarSelection({false, true});
}

TEST(DistanceDispatch, BothFeaturesSelectCompiledBackend) {
    // 模拟特性时只比较函数地址，不调用潜在不受本机支持的指令。
    const ann::CpuFeatures both{true, true};
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    EXPECT_EQ(ann::detail::SelectDistFnForFeatures(ann::Metric::L2, both), ann::L2SqrAVX2Four);
    EXPECT_EQ(ann::detail::SelectDistFnForFeatures(ann::Metric::InnerProduct, both),
              ann::InnerProductAVX2Four);
    EXPECT_EQ(ann::detail::SelectDistFnForFeatures(ann::Metric::Cosine, both),
              ann::InnerProductAVX2Four);
#else
    ExpectScalarSelection(both);
#endif
}

TEST(DistanceDispatch, AutoSelectionMatchesActualFeatures) {
    const auto& features = ann::CpuFeatures::Get();
    const bool simd = features.avx2 && features.fma;
    EXPECT_EQ(ann::SelectDistFn(ann::Metric::L2), simd ? ann::L2SqrAVX2Four : ann::L2SqrScalar);
    EXPECT_EQ(ann::SelectDistFn(ann::Metric::InnerProduct),
              simd ? ann::InnerProductAVX2Four : ann::InnerProductScalar);
    EXPECT_EQ(ann::SelectDistFn(ann::Metric::Cosine), ann::SelectDistFn(ann::Metric::InnerProduct));
    EXPECT_EQ(ann::SelectDistFn(static_cast<ann::Metric>(255)),
              ann::SelectDistFn(ann::Metric::InnerProduct));
}

TEST(DistanceDispatch, SelectedFunctionsMatchIndependentDoubleReference) {
    const auto l2_fn = ann::SelectDistFn(ann::Metric::L2);
    const auto ip_fn = ann::SelectDistFn(ann::Metric::InnerProduct);
    const auto cosine_fn = ann::SelectDistFn(ann::Metric::Cosine);
    for (const size_t dim : {0u, 1u, 7u, 8u, 31u, 32u, 33u, 127u, 128u, 129u, 960u}) {
        SCOPED_TRACE(dim);
        // 偏移一个 float，并且仅留实际长度，配合 ASan 检查任意维度与尾部。
        std::vector<float> a(dim + 1);
        std::vector<float> b(dim + 1);
        double l2 = 0.0;
        double ip = 0.0;
        double scale = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            a[i + 1] = static_cast<float>(static_cast<int>(i % 17) - 8) / 7.0f;
            b[i + 1] = static_cast<float>(static_cast<int>(i % 23) - 11) / 11.0f;
            const double x = static_cast<double>(a[i + 1]);
            const double y = static_cast<double>(b[i + 1]);
            l2 += (x - y) * (x - y);
            ip += x * y;
            scale += std::abs(x * y);
        }
        EXPECT_NEAR(static_cast<double>(l2_fn(a.data() + 1, b.data() + 1, dim)),
                    l2,
                    2e-5 * std::max(1.0, l2));
        EXPECT_NEAR(static_cast<double>(ip_fn(a.data() + 1, b.data() + 1, dim)),
                    -ip,
                    2e-5 * std::max(1.0, scale));
        EXPECT_NEAR(static_cast<double>(cosine_fn(a.data() + 1, b.data() + 1, dim)),
                    -ip,
                    2e-5 * std::max(1.0, scale));
    }
}

TEST(DistanceDispatch, FlatIndexAutomaticallyUsesSelectedDistance) {
    for (const size_t dim : {1u, 7u, 32u, 127u, 128u, 129u}) {
        SCOPED_TRACE(dim);
        const float values[4] = {0.0f, 1.0f, 3.0f, 10.0f};
        std::vector<float> base(4 * dim);
        for (size_t i = 0; i < 4; ++i) {
            std::fill_n(base.data() + i * dim, dim, values[i]);
        }
        std::vector<float> query(dim, 0.9f);
        float dists[2]{};
        ann::idx_t ids[2]{};
        ann::FlatIndex l2_index(dim, ann::Metric::L2);
        l2_index.Add(4, base.data());
        l2_index.Search(query.data(), 2, {dists, ids, 2});
        EXPECT_EQ(ids[0], 1u);
        EXPECT_EQ(ids[1], 0u);
        EXPECT_NEAR(dists[0], 0.01f * static_cast<float>(dim), 1e-4f * static_cast<float>(dim));
        EXPECT_NEAR(dists[1], 0.81f * static_cast<float>(dim), 1e-4f * static_cast<float>(dim));

        ann::FlatIndex ip_index(dim, ann::Metric::InnerProduct);
        ip_index.Add(4, base.data());
        ip_index.Search(query.data(), 2, {dists, ids, 2});
        EXPECT_EQ(ids[0], 3u);
        EXPECT_EQ(ids[1], 2u);
        EXPECT_NEAR(dists[0], -9.0f * static_cast<float>(dim), 1e-4f * static_cast<float>(dim));
        EXPECT_NEAR(dists[1], -2.7f * static_cast<float>(dim), 1e-4f * static_cast<float>(dim));
    }
}

}  // namespace
