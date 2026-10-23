#include "hnsw_search.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>

namespace {

using ann::detail::HNSWCandidate;
using ann::detail::HNSWGraphStorage;
using ann::detail::HNSWGreedyDescend;
using ann::detail::HNSWSearchLayer;
using ann::detail::HNSWSearchWorkspace;

void Link(HNSWGraphStorage& graph,
          ann::idx_t node,
          uint8_t level,
          std::initializer_list<ann::idx_t> neighbors) {
    graph.SetNeighbors(node, level, {neighbors.begin(), neighbors.size()});
}

HNSWGraphStorage CompleteGraph(std::span<const float> values, uint8_t level = 0) {
    HNSWGraphStorage graph(
        1, std::max(values.size(), size_t{1}), std::max(values.size(), size_t{1}));
    for (const float value : values) {
        graph.Append({&value, 1}, level);
    }
    for (ann::idx_t i = 0; i < values.size(); ++i) {
        std::vector<ann::idx_t> neighbors;
        for (ann::idx_t j = 0; j < values.size(); ++j) {
            if (i != j) {
                neighbors.push_back(j);
            }
        }
        for (int layer = 0; layer <= level; ++layer) {
            graph.SetNeighbors(i, static_cast<uint8_t>(layer), neighbors);
        }
    }
    return graph;
}

std::vector<ann::idx_t> Ids(std::span<const HNSWCandidate> candidates) {
    std::vector<ann::idx_t> ids;
    for (const auto& [distance, id] : candidates) {
        static_cast<void>(distance);
        ids.push_back(id);
    }
    return ids;
}

TEST(HNSWSearch, GreedyConvergesAlongImprovingChain) {
    HNSWGraphStorage graph(1, 2, 2);
    for (const float value : {9.0f, 6.0f, 3.0f, 0.0f}) {
        graph.Append({&value, 1}, 1);
    }
    Link(graph, 0, 1, {1});
    Link(graph, 1, 1, {0, 2});
    Link(graph, 2, 1, {1, 3});
    Link(graph, 3, 1, {2});
    const std::array<float, 1> query{0};
    EXPECT_EQ(HNSWGreedyDescend(graph, query, 0, 1, ann::L2SqrScalar), 3u);
    EXPECT_EQ(HNSWGreedyDescend(graph, query, 3, 1, ann::L2SqrScalar), 3u);
}

TEST(HNSWSearch, GreedyStopsAtEqualDistancePlateau) {
    HNSWGraphStorage graph(1, 2, 2);
    for (const float value : {1.0f, -1.0f, 0.0f}) {
        graph.Append({&value, 1}, 0);
    }
    Link(graph, 0, 0, {1});
    Link(graph, 1, 0, {0, 2});
    const std::array<float, 1> query{0};
    EXPECT_EQ(HNSWGreedyDescend(graph, query, 0, 0, ann::L2SqrScalar), 0u);
}

TEST(HNSWSearch, GreedyUsesOnlyRequestedLayerAndBestNeighbor) {
    HNSWGraphStorage graph(1, 3, 3);
    for (const float value : {10.0f, 4.0f, 1.0f, 0.0f}) {
        graph.Append({&value, 1}, 1);
    }
    Link(graph, 0, 1, {1, 2});
    Link(graph, 0, 0, {3});
    const std::array<float, 1> query{0};
    EXPECT_EQ(HNSWGreedyDescend(graph, query, 0, 1, ann::L2SqrScalar), 2u);
    EXPECT_EQ(HNSWGreedyDescend(graph, query, 0, 0, ann::L2SqrScalar), 3u);
}

TEST(HNSWSearch, NegativeInnerProductKeepsSmallerDistanceConvention) {
    const std::array<float, 4> values{-2, 1, 3, 8};
    const auto graph = CompleteGraph(values);
    const std::array<float, 1> query{2};
    EXPECT_EQ(HNSWGreedyDescend(graph, query, 0, 0, ann::InnerProductScalar), 3u);
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), 2);
    const auto result = HNSWSearchLayer(graph, query, 0, 2, 0, ann::InnerProductScalar, workspace);
    EXPECT_EQ(Ids(result), (std::vector<ann::idx_t>{3, 2}));
    EXPECT_FLOAT_EQ(result[0].first, -16.0f);
}

TEST(HNSWSearch, CompleteTwoDimensionalGraphMatchesIndependentDoubleReference) {
    HNSWGraphStorage graph(2, 10, 10);
    for (size_t i = 0; i < 10; ++i) {
        const std::array<float, 2> vector{static_cast<float>(i), static_cast<float>((i * 7) % 5)};
        graph.Append(vector, 1);
    }
    for (ann::idx_t i = 0; i < 10; ++i) {
        std::vector<ann::idx_t> neighbors;
        for (ann::idx_t j = 0; j < 10; ++j) {
            if (i != j) {
                neighbors.push_back(j);
            }
        }
        graph.SetNeighbors(i, 0, neighbors);
        graph.SetNeighbors(i, 1, neighbors);
    }
    const std::array<float, 2> query{2.6f, 2.2f};
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), 4);
    for (const auto metric : {ann::Metric::L2, ann::Metric::InnerProduct}) {
        std::vector<std::pair<double, ann::idx_t>> reference;
        for (ann::idx_t i = 0; i < 10; ++i) {
            double distance = 0;
            for (size_t d = 0; d < 2; ++d) {
                const double x = query[d];
                const double y = graph.Vector(i)[d];
                distance += metric == ann::Metric::L2 ? (x - y) * (x - y) : -x * y;
            }
            reference.emplace_back(distance, i);
        }
        std::sort(reference.begin(), reference.end());
        for (const int level : {0, 1}) {
            const auto result =
                HNSWSearchLayer(graph, query, 9, 4, level, ann::SelectDistFn(metric), workspace);
            ASSERT_EQ(result.size(), 4u);
            for (size_t i = 0; i < result.size(); ++i) {
                EXPECT_EQ(result[i].second, reference[i].second);
                EXPECT_NEAR(result[i].first, reference[i].first, 1e-4);
            }
        }
    }
}

TEST(HNSWSearch, WideEfTraversesChainAndClampsToGraphSize) {
    HNSWGraphStorage graph(1, 2, 2);
    for (const float value : {8.0f, 6.0f, 4.0f, 2.0f, 0.0f}) {
        graph.Append({&value, 1}, 0);
    }
    for (ann::idx_t i = 0; i < 4; ++i) {
        const std::array<ann::idx_t, 1> next{i + 1};
        graph.SetNeighbors(i, 0, next);
    }
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), std::numeric_limits<size_t>::max());
    const std::array<float, 1> query{0};
    const auto result = HNSWSearchLayer(
        graph, query, 0, std::numeric_limits<size_t>::max(), 0, ann::L2SqrScalar, workspace);
    EXPECT_EQ(Ids(result), (std::vector<ann::idx_t>{4, 3, 2, 1, 0}));
}

std::array<size_t, 5> distance_calls{};

ann::dist_t CountedL2(const float* query, const float* vector, size_t dim) {
    ++distance_calls[static_cast<size_t>(vector[0])];
    return ann::L2SqrScalar(query, vector, dim);
}

TEST(HNSWSearch, CyclesAndMultiplePathsEvaluateEachNodeOnlyOnce) {
    const std::array<float, 5> values{0, 1, 2, 3, 4};
    const auto graph = CompleteGraph(values);
    distance_calls.fill(0);
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), graph.Size());
    const std::array<float, 1> query{0};
    EXPECT_EQ(HNSWSearchLayer(graph, query, 0, graph.Size(), 0, CountedL2, workspace).size(), 5u);
    for (const auto calls : distance_calls) {
        EXPECT_EQ(calls, 1u);
    }
}

TEST(HNSWSearch, DisconnectedGraphReturnsOnlyReachableCandidates) {
    HNSWGraphStorage graph(1, 2, 2);
    for (const float value : {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}) {
        graph.Append({&value, 1}, 0);
    }
    Link(graph, 0, 0, {1});
    Link(graph, 1, 0, {0, 2});
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), 10);
    const std::array<float, 1> query{4};
    const auto result = HNSWSearchLayer(graph, query, 0, 10, 0, ann::L2SqrScalar, workspace);
    EXPECT_EQ(Ids(result), (std::vector<ann::idx_t>{2, 1, 0}));
}

TEST(HNSWSearch, ClosestFrontierFirstAndWiderEfCanCrossFartherBridge) {
    HNSWGraphStorage graph(1, 2, 2);
    for (const float value : {10.0f, 3.0f, 1.0f, 0.0f, 2.0f}) {
        graph.Append({&value, 1}, 0);
    }
    Link(graph, 0, 0, {1, 2});
    Link(graph, 1, 0, {3});
    Link(graph, 2, 0, {4});
    const std::array<float, 1> query{0};
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), 3);
    EXPECT_EQ(Ids(HNSWSearchLayer(graph, query, 0, 2, 0, ann::L2SqrScalar, workspace)),
              (std::vector<ann::idx_t>{2, 4}));
    EXPECT_EQ(Ids(HNSWSearchLayer(graph, query, 0, 3, 0, ann::L2SqrScalar, workspace)),
              (std::vector<ann::idx_t>{3, 2, 4}));
}

TEST(HNSWSearch, EqualDistancesProduceDistinctIdsInDeterministicOrder) {
    const std::array<float, 5> values{1, 1, 1, 1, 1};
    const auto graph = CompleteGraph(values);
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), 2);
    const std::array<float, 1> query{0};
    EXPECT_EQ(Ids(HNSWSearchLayer(graph, query, 4, 2, 0, ann::L2SqrScalar, workspace)),
              (std::vector<ann::idx_t>{0, 1}));
}

TEST(HNSWSearch, EmptyGraphAndZeroEfReturnEmptyOutput) {
    HNSWGraphStorage empty(1, 2, 2);
    const std::array<float, 1> query{0};
    HNSWSearchWorkspace workspace;
    EXPECT_EQ(HNSWGreedyDescend(empty, query, ann::kInvalidIdx, 0, ann::L2SqrScalar),
              ann::kInvalidIdx);
    EXPECT_TRUE(HNSWSearchLayer(empty, query, ann::kInvalidIdx, 10, 0, ann::L2SqrScalar, workspace)
                    .empty());
    const auto graph = CompleteGraph(query);
    workspace.results.emplace_back(99.0f, 99);
    EXPECT_TRUE(
        HNSWSearchLayer(graph, query, ann::kInvalidIdx, 0, 0, ann::L2SqrScalar, workspace).empty());
}

TEST(HNSWSearch, InvalidInputsAreRejected) {
    const std::array<float, 1> query{0};
    const auto graph = CompleteGraph(query);
    HNSWSearchWorkspace workspace;
    workspace.Prepare(1, 1);
    EXPECT_THROW(HNSWGreedyDescend(graph, {}, 0, 0, ann::L2SqrScalar), std::invalid_argument);
    EXPECT_THROW(HNSWGreedyDescend(graph, query, 0, -1, ann::L2SqrScalar), std::out_of_range);
    EXPECT_THROW(HNSWGreedyDescend(graph, query, 0, 256, ann::L2SqrScalar), std::out_of_range);
    EXPECT_THROW(HNSWGreedyDescend(graph, query, 1, 0, ann::L2SqrScalar), std::out_of_range);
    EXPECT_THROW(HNSWGreedyDescend(graph, query, 0, 1, ann::L2SqrScalar), std::out_of_range);
    EXPECT_THROW(HNSWGreedyDescend(graph, query, 0, 0, nullptr), std::invalid_argument);
    EXPECT_THROW(HNSWSearchLayer(graph, {}, 0, 1, 0, ann::L2SqrScalar, workspace),
                 std::invalid_argument);
    EXPECT_THROW(HNSWSearchLayer(graph, query, 0, 1, -1, ann::L2SqrScalar, workspace),
                 std::out_of_range);
    EXPECT_THROW(HNSWSearchLayer(graph, query, 0, 1, 256, ann::L2SqrScalar, workspace),
                 std::out_of_range);
    EXPECT_THROW(HNSWSearchLayer(graph, query, 1, 1, 0, ann::L2SqrScalar, workspace),
                 std::out_of_range);
    EXPECT_THROW(HNSWSearchLayer(graph, query, 0, 1, 1, ann::L2SqrScalar, workspace),
                 std::out_of_range);
    EXPECT_THROW(HNSWSearchLayer(graph, query, 0, 1, 0, nullptr, workspace), std::invalid_argument);
}

TEST(HNSWSearch, MissingOrUndersizedWorkspaceIsRejected) {
    const std::array<float, 3> values{0, 1, 2};
    const auto graph = CompleteGraph(values);
    const std::array<float, 1> query{0};
    HNSWSearchWorkspace workspace;
    EXPECT_THROW(HNSWSearchLayer(graph, query, 0, 2, 0, ann::L2SqrScalar, workspace),
                 std::length_error);
    workspace.Prepare(2, 2);
    EXPECT_THROW(HNSWSearchLayer(graph, query, 0, 2, 0, ann::L2SqrScalar, workspace),
                 std::length_error);
    workspace.Prepare(3, 1);
    // 前一次预留的 capacity 可以更大；显式收缩结果缓冲后检查不足路径。
    std::vector<HNSWCandidate>().swap(workspace.results);
    EXPECT_THROW(HNSWSearchLayer(graph, query, 0, 2, 0, ann::L2SqrScalar, workspace),
                 std::length_error);
}

TEST(HNSWSearch, WorkspaceReuseKeepsBuffersStableAndClearsVisitedState) {
    const std::array<float, 6> values{0, 1, 2, 3, 4, 5};
    const auto graph = CompleteGraph(values);
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), 2);
    const auto* frontier = workspace.frontier.data();
    const auto* results = workspace.results.data();
    const auto* visited = workspace.visited.data();
    const auto bytes = workspace.MemoryUsage();
    for (const float q : {0.0f, 5.0f, 0.0f, 5.0f}) {
        const std::array<float, 1> query{q};
        const auto result = HNSWSearchLayer(graph, query, 0, 2, 0, ann::L2SqrScalar, workspace);
        EXPECT_EQ(result.front().second, q < 1.0f ? 0u : 5u);
        EXPECT_EQ(workspace.frontier.data(), frontier);
        EXPECT_EQ(workspace.results.data(), results);
        EXPECT_EQ(workspace.visited.data(), visited);
        EXPECT_EQ(workspace.MemoryUsage(), bytes);
    }
}

TEST(HNSWSearch, SearchesPreserveGraphDataEdgesAndMetadata) {
    const std::array<float, 4> values{3, 1, 2, 0};
    const auto graph = CompleteGraph(values, 1);
    const auto bytes = graph.MemoryUsage();
    const auto entry = graph.EntryPoint();
    HNSWSearchWorkspace workspace;
    workspace.Prepare(graph.Size(), graph.Size());
    const std::array<float, 1> query{0};
    HNSWGreedyDescend(graph, query, entry, 1, ann::L2SqrScalar);
    HNSWSearchLayer(graph, query, entry, graph.Size(), 0, ann::L2SqrScalar, workspace);
    EXPECT_EQ(graph.MemoryUsage(), bytes);
    EXPECT_EQ(graph.EntryPoint(), entry);
    EXPECT_EQ(graph.MaxLevel(), 1);
    for (ann::idx_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(graph.Vector(i).front(), values[i]);
        EXPECT_EQ(graph.Level(i), 1u);
        for (const uint8_t level : {0, 1}) {
            const auto neighbors = graph.Neighbors(i, level);
            ASSERT_EQ(neighbors.size(), 3u);
            size_t offset = 0;
            for (ann::idx_t j = 0; j < values.size(); ++j) {
                if (i != j) {
                    EXPECT_EQ(neighbors[offset++], j);
                }
            }
        }
    }
}

TEST(HNSWSearch, IndependentWorkspacesDoNotOverwriteEachOthersResults) {
    const std::array<float, 3> values{0, 1, 2};
    const auto graph = CompleteGraph(values);
    HNSWSearchWorkspace first;
    HNSWSearchWorkspace second;
    first.Prepare(3, 2);
    second.Prepare(3, 2);
    const std::array<float, 1> left{0};
    const std::array<float, 1> right{2};
    const auto left_result = HNSWSearchLayer(graph, left, 0, 2, 0, ann::L2SqrScalar, first);
    const auto right_result = HNSWSearchLayer(graph, right, 0, 2, 0, ann::L2SqrScalar, second);
    EXPECT_EQ(Ids(left_result), (std::vector<ann::idx_t>{0, 1}));
    EXPECT_EQ(Ids(right_result), (std::vector<ann::idx_t>{2, 1}));
}

ann::dist_t FailsOnSecondNode(const float* query, const float* vector, size_t dim) {
    return static_cast<size_t>(vector[0]) == 1 ? std::numeric_limits<float>::quiet_NaN()
                                               : ann::L2SqrScalar(query, vector, dim);
}

TEST(HNSWSearch, NonfiniteDistanceIsRejectedAndWorkspaceCanBeReused) {
    const std::array<float, 3> values{0, 1, 2};
    const auto graph = CompleteGraph(values);
    HNSWSearchWorkspace workspace;
    workspace.Prepare(3, 2);
    const std::array<float, 1> query{0};
    EXPECT_THROW(HNSWGreedyDescend(graph, query, 0, 0, FailsOnSecondNode), std::invalid_argument);
    EXPECT_THROW(HNSWSearchLayer(graph, query, 0, 2, 0, FailsOnSecondNode, workspace),
                 std::invalid_argument);
    EXPECT_EQ(Ids(HNSWSearchLayer(graph, query, 0, 2, 0, ann::L2SqrScalar, workspace)),
              (std::vector<ann::idx_t>{0, 1}));
    const std::array<float, 1> infinite{std::numeric_limits<float>::infinity()};
    EXPECT_THROW(HNSWSearchLayer(graph, infinite, 0, 2, 0, ann::L2SqrScalar, workspace),
                 std::invalid_argument);
}

TEST(HNSWSearch, WorkspaceRejectsUnaddressableNodeCount) {
    HNSWSearchWorkspace workspace;
    EXPECT_EQ(workspace.MemoryUsage(), sizeof(workspace));
    EXPECT_THROW(workspace.Prepare(static_cast<size_t>(ann::kInvalidIdx) + 1, 1),
                 std::length_error);
}

}  // namespace
