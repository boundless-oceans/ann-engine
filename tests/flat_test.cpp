// 用独立手算结果验证精确检索，不用 FlatIndex 自己生成期望值。
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>

#include "ann/flat.h"

namespace {

TEST(FlatIndex, L2HandComputed) {
    const float base[] = {0.0f, 0.0f, 1.0f, 0.0f, 10.0f, 0.0f};
    const float query[] = {0.9f, 0.0f};
    ann::FlatIndex index(2, ann::Metric::L2);
    index.Add(3, base);
    float dists[2]{};
    ann::idx_t ids[2]{};

    // 经 const 引用调用，确保测试走的是只读查询接口。
    const ann::FlatIndex& reader = index;
    reader.Search(query, 2, {dists, ids, 2});

    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 0u);
    EXPECT_NEAR(dists[0], 0.01f, 1e-5f);
    EXPECT_NEAR(dists[1], 0.81f, 1e-5f);
}

TEST(FlatIndex, ReplacesHeapRootAndSortsResults) {
    // 最近点放在最后，强制反复替换堆顶；距离分别为 25、16、9、4、1。
    const float base[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    const float query[] = {0.0f};
    ann::FlatIndex index(1, ann::Metric::L2);
    index.Add(5, base);
    float dists[3]{};
    ann::idx_t ids[3]{};

    index.Search(query, 3, {dists, ids, 3});

    EXPECT_EQ(ids[0], 4u);
    EXPECT_EQ(ids[1], 3u);
    EXPECT_EQ(ids[2], 2u);
    EXPECT_NEAR(dists[0], 1.0f, 1e-5f);
    EXPECT_NEAR(dists[1], 4.0f, 1e-5f);
    EXPECT_NEAR(dists[2], 9.0f, 1e-5f);
}

TEST(FlatIndex, TopOneFindsLaterCandidate) {
    const float base[] = {5.0f, 0.0f, 2.0f};
    const float query[] = {1.9f};
    ann::FlatIndex index(1, ann::Metric::L2);
    index.Add(3, base);
    float dist = -1.0f;
    ann::idx_t id = ann::kInvalidIdx;

    index.Search(query, 1, {&dist, &id, 1});

    EXPECT_EQ(id, 2u);
    EXPECT_NEAR(dist, 0.01f, 1e-5f);
}

TEST(FlatIndex, TooFewVectorsFillSentinels) {
    const float base[] = {3.0f, 1.0f};
    const float query[] = {0.0f};
    ann::FlatIndex index(1, ann::Metric::L2);
    index.Add(2, base);
    float dists[4]{};
    ann::idx_t ids[4]{};

    index.Search(query, 4, {dists, ids, 4});

    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 0u);
    EXPECT_NEAR(dists[0], 1.0f, 1e-5f);
    EXPECT_NEAR(dists[1], 9.0f, 1e-5f);
    for (size_t i = 2; i < 4; ++i) {
        EXPECT_EQ(ids[i], ann::kInvalidIdx);
        EXPECT_FLOAT_EQ(dists[i], std::numeric_limits<ann::dist_t>::max());
    }
}

TEST(FlatIndex, EmptyIndexFillsSentinels) {
    const float query[] = {1.0f, 2.0f};
    const ann::FlatIndex index(2, ann::Metric::L2);
    float dists[2]{};
    ann::idx_t ids[2]{};

    index.Search(query, 2, {dists, ids, 2});

    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(ids[i], ann::kInvalidIdx);
        EXPECT_FLOAT_EQ(dists[i], std::numeric_limits<ann::dist_t>::max());
    }
}

TEST(FlatIndex, ZeroRequestedResultsLeaveOutputUntouched) {
    const float base[] = {1.0f};
    const float query[] = {0.0f};
    ann::FlatIndex index(1, ann::Metric::L2);
    index.Add(1, base);
    float dist = -7.0f;
    ann::idx_t id = 42;

    index.Search(query, 0, {&dist, &id, 1});
    index.Search(query, 1, {&dist, &id, 0});

    EXPECT_FLOAT_EQ(dist, -7.0f);
    EXPECT_EQ(id, 42u);
}

TEST(FlatIndex, OutputCapacityLimitsWrites) {
    const float base[] = {3.0f, 1.0f, 2.0f};
    const float query[] = {0.0f};
    ann::FlatIndex index(1, ann::Metric::L2);
    index.Add(3, base);
    float dists[] = {-7.0f, -7.0f, -7.0f};
    ann::idx_t ids[] = {42, 42, 42};

    // 第三个元素作保护值，检查是否写出了 ResultView 声明的容量。
    index.Search(query, 3, {dists, ids, 2});

    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 2u);
    EXPECT_NEAR(dists[0], 1.0f, 1e-5f);
    EXPECT_NEAR(dists[1], 4.0f, 1e-5f);
    EXPECT_EQ(ids[2], 42u);
    EXPECT_FLOAT_EQ(dists[2], -7.0f);
}

TEST(FlatIndex, RequestedCountLimitsWrites) {
    const float base[] = {3.0f, 1.0f, 2.0f};
    const float query[] = {0.0f};
    ann::FlatIndex index(1, ann::Metric::L2);
    index.Add(3, base);
    float dists[] = {-7.0f, -7.0f, -7.0f};
    ann::idx_t ids[] = {42, 42, 42};

    index.Search(query, 1, {dists, ids, 3});

    EXPECT_EQ(ids[0], 1u);
    EXPECT_NEAR(dists[0], 1.0f, 1e-5f);
    for (size_t i = 1; i < 3; ++i) {
        EXPECT_EQ(ids[i], 42u);
        EXPECT_FLOAT_EQ(dists[i], -7.0f);
    }
}

TEST(FlatIndex, MultipleAddsPreserveIdsAndCopyInput) {
    float first[] = {5.0f, 0.0f};
    const float second[] = {1.0f, 0.0f, 3.0f, 0.0f};
    const float query[] = {0.0f, 0.0f};
    ann::FlatIndex index(2, ann::Metric::L2);
    index.Add(1, first);
    first[0] = 100.0f;
    index.Add(2, second);
    float dists[3]{};
    ann::idx_t ids[3]{};

    index.Search(query, 3, {dists, ids, 3});

    EXPECT_EQ(index.Size(), 3u);
    EXPECT_EQ(index.Dim(), 2u);
    EXPECT_EQ(index.GetMetric(), ann::Metric::L2);
    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 2u);
    EXPECT_EQ(ids[2], 0u);
    EXPECT_NEAR(dists[0], 1.0f, 1e-5f);
    EXPECT_NEAR(dists[1], 9.0f, 1e-5f);
    EXPECT_NEAR(dists[2], 25.0f, 1e-5f);
}

TEST(FlatIndex, InnerProductReturnsLargestProductsFirst) {
    const float base[] = {1.0f, 0.0f, -3.0f, 0.0f, 4.0f, 0.0f};
    const float query[] = {2.0f, 0.0f};
    ann::FlatIndex index(2, ann::Metric::InnerProduct);
    index.Add(3, base);
    float dists[2]{};
    ann::idx_t ids[2]{};

    index.Search(query, 2, {dists, ids, 2});

    EXPECT_EQ(ids[0], 2u);
    EXPECT_EQ(ids[1], 0u);
    EXPECT_NEAR(dists[0], -8.0f, 1e-5f);
    EXPECT_NEAR(dists[1], -2.0f, 1e-5f);
}

TEST(FlatIndex, EqualDistancesReturnDistinctIds) {
    const float base[] = {1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f};
    const float query[] = {0.0f, 0.0f};
    ann::FlatIndex index(2, ann::Metric::L2);
    index.Add(3, base);
    float dists[3]{};
    std::array<ann::idx_t, 3> ids{};

    index.Search(query, 3, {dists, ids.data(), 3});

    // 相同距离不约定 ID 的先后，只验证结果没有丢失或重复。
    for (float dist : dists) {
        EXPECT_NEAR(dist, 1.0f, 1e-5f);
    }
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids[0], 0u);
    EXPECT_EQ(ids[1], 1u);
    EXPECT_EQ(ids[2], 2u);
}

TEST(FlatIndex, SearchBatchUsesSeparateOutputRows) {
    const float base[] = {0.0f, 0.0f, 1.0f, 0.0f, 10.0f, 0.0f};
    const float queries[] = {0.9f, 0.0f, 9.0f, 0.0f};
    ann::FlatIndex index(2, ann::Metric::L2);
    index.Add(3, base);
    float dists[4]{};
    ann::idx_t ids[4]{};

    index.SearchBatch(queries, 2, 2, {dists, ids, 2});

    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 0u);
    EXPECT_EQ(ids[2], 2u);
    EXPECT_EQ(ids[3], 1u);
    EXPECT_NEAR(dists[0], 0.01f, 1e-5f);
    EXPECT_NEAR(dists[1], 0.81f, 1e-5f);
    EXPECT_NEAR(dists[2], 1.0f, 1e-5f);
    EXPECT_NEAR(dists[3], 64.0f, 1e-5f);
}

}  // namespace
