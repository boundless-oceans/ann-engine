#include "hnsw_neighbors.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>

#include "ann/distance.h"
#include "ann/hnsw.h"

namespace {

using ann::detail::HNSWCandidate;
using ann::detail::HNSWSelectNeighbors;

TEST(HNSWNeighbors, KeepsFartherOppositeDirectionInsteadOfRedundantNearPoint) {
    // 中心(0,0)：A(1,0)、B(2,0) 同向，C(-3,0) 更远但提供另一个方向。
    const float vectors[][2] = {{1, 0}, {2, 0}, {-3, 0}};
    const std::array<HNSWCandidate, 3> candidates{{{1, 0}, {4, 1}, {9, 2}}};
    std::array<HNSWCandidate, 2> selected;
    const auto count = HNSWSelectNeighbors(candidates, selected, [&](auto a, auto b) {
        return ann::L2SqrScalar(vectors[a], vectors[b], 2);
    });
    ASSERT_EQ(count, 2u);
    EXPECT_EQ(selected[0].second, 0u);
    EXPECT_EQ(selected[1].second, 2u);
    EXPECT_EQ(candidates[1].second, 1u);
}

TEST(HNSWNeighbors, CanReturnFewerThanCapacityWithoutRefillingRejectedPoints) {
    const float positions[] = {1, 2, 3, 4};
    const std::array<HNSWCandidate, 4> candidates{{{1, 0}, {4, 1}, {9, 2}, {16, 3}}};
    std::array<HNSWCandidate, 3> selected;
    const auto count = HNSWSelectNeighbors(candidates, selected, [&](auto a, auto b) {
        const auto d = positions[a] - positions[b];
        return d * d;
    });
    ASSERT_EQ(count, 1u);
    EXPECT_EQ(selected[0].second, 0u);
}

TEST(HNSWNeighbors, EqualDistanceIsAcceptedAndIdOrderIsRetained) {
    const std::array<HNSWCandidate, 3> candidates{{{1, 3}, {1, 5}, {1, 7}}};
    std::array<HNSWCandidate, 2> selected;
    size_t calls = 0;
    const auto count = HNSWSelectNeighbors(candidates, selected, [&](auto, auto) {
        ++calls;
        return 1.0f;
    });
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(selected[0].second, 3u);
    EXPECT_EQ(selected[1].second, 5u);
}

TEST(HNSWNeighbors, FitsCapacityKeepsAllWithoutPairEvaluation) {
    const std::array<HNSWCandidate, 2> candidates{{{1, 0}, {4, 1}}};
    std::array<HNSWCandidate, 3> selected;
    const auto count = HNSWSelectNeighbors(candidates, selected, [](auto, auto) -> float {
        ADD_FAILURE() << "within-capacity candidates must be retained";
        return 0;
    });
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(selected[1], candidates[1]);
}

TEST(HNSWNeighbors, EmptyInputsAndOutputDoNotEvaluatePairs) {
    std::array<HNSWCandidate, 1> selected;
    const std::array<HNSWCandidate, 1> candidates{{{1, 0}}};
    auto fail = [](auto, auto) -> float {
        ADD_FAILURE() << "empty selection must not calculate pairs";
        return 0;
    };
    EXPECT_EQ(HNSWSelectNeighbors({}, selected, fail), 0u);
    EXPECT_EQ(HNSWSelectNeighbors(candidates, {}, fail), 0u);
}

TEST(HNSWNeighbors, ChecksEverySelectedNeighborForCoverage) {
    // C 只被第二个已选点覆盖，不能只和最近点比较。
    const std::array<HNSWCandidate, 4> candidates{{{1, 0}, {2, 1}, {3, 2}, {4, 3}}};
    std::array<HNSWCandidate, 3> selected;
    const auto count = HNSWSelectNeighbors(
        candidates, selected, [](auto a, auto b) { return a == 2 && b == 1 ? 1.0f : 100.0f; });
    ASSERT_EQ(count, 3u);
    EXPECT_EQ(selected[2].second, 3u);
}

TEST(HNSWNeighbors, RejectsNonFiniteCenterAndPairDistances) {
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto inf = std::numeric_limits<float>::infinity();
    std::array<HNSWCandidate, 2> selected;
    std::array<HNSWCandidate, 3> candidates{{{1, 0}, {2, 1}, {3, 2}}};
    EXPECT_THROW(HNSWSelectNeighbors(candidates, selected, [inf](auto, auto) { return inf; }),
                 std::invalid_argument);
    candidates[1].first = nan;
    EXPECT_THROW(HNSWSelectNeighbors(candidates, selected, [](auto, auto) { return 1.0f; }),
                 std::invalid_argument);
}

TEST(HNSWNeighbors, NegativeInnerProductUsesLowerIsBetterComparison) {
    const std::array<HNSWCandidate, 3> candidates{{{-3, 0}, {-2, 1}, {-1, 2}}};
    std::array<HNSWCandidate, 2> selected;
    const auto count = HNSWSelectNeighbors(
        candidates, selected, [](auto a, auto) { return a == 1 ? -4.0f : 0.0f; });
    ASSERT_EQ(count, 2u);
    EXPECT_EQ(selected[1].second, 2u);
}

TEST(HNSWNeighbors, PendingNodeCanParticipateBeforeGraphAppend) {
    ann::detail::HNSWGraphStorage graph(1, 2, 4);
    const float a[] = {1};
    const float b[] = {2};
    const float pending[] = {-3};
    graph.Append(a, 0);
    graph.Append(b, 0);
    const std::array<HNSWCandidate, 3> candidates{{{1, 0}, {4, 1}, {9, 2}}};
    std::array<HNSWCandidate, 2> selected;
    const auto count = HNSWSelectNeighbors(candidates, selected, [&](auto x, auto y) {
        const auto* xv = x == 2 ? pending : graph.Vector(x).data();
        const auto* yv = y == 2 ? pending : graph.Vector(y).data();
        return ann::L2SqrScalar(xv, yv, 1);
    });
    ASSERT_EQ(count, 2u);
    EXPECT_EQ(selected[1].second, 2u);
    EXPECT_EQ(graph.Size(), 2u);
}

TEST(HNSWIndex, SmallMPruningPreservesLineNavigationAndDegreeBounds) {
    ann::HNSWParams params;
    params.M = 2;
    params.Mmax0 = 4;
    params.ef_construction = 32;
    params.ef_search = 32;
    ann::HNSWIndex index(1, ann::Metric::L2, params);
    std::vector<float> points;
    for (size_t i = 0; i < 40; ++i) {
        points.push_back(static_cast<float>(i));
    }
    index.Add(points.size(), points.data());
    for (size_t i = 0; i < points.size(); ++i) {
        float distances[3];
        ann::idx_t ids[3];
        index.Search(&points[i], 3, {distances, ids, 3});
        EXPECT_EQ(ids[0], i);
        EXPECT_EQ(distances[0], 0);
        EXPECT_NE(ids[1], ann::kInvalidIdx);
        EXPECT_NE(ids[2], ann::kInvalidIdx);
    }
    const auto stats = index.GetStats();
    EXPECT_LE(stats.edges_level0, points.size() * params.Mmax0);
    EXPECT_LE(stats.edges_upper, points.size() * static_cast<size_t>(stats.max_level) * params.M);
}

}  // namespace
