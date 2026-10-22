#include "hnsw_graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace {

using ann::detail::HNSWGraphStorage;

std::vector<ann::idx_t> NeighborIds(const HNSWGraphStorage& graph, ann::idx_t node, uint8_t level) {
    const auto neighbors = graph.Neighbors(node, level);
    return {neighbors.begin(), neighbors.end()};
}

TEST(HNSWGraphStorage, EmptyGraphAndParameterValidation) {
    HNSWGraphStorage graph(2, 2, 4);
    EXPECT_EQ(graph.Size(), 0u);
    EXPECT_EQ(graph.Dim(), 2u);
    EXPECT_EQ(graph.EntryPoint(), ann::kInvalidIdx);
    EXPECT_EQ(graph.MaxLevel(), -1);
    EXPECT_EQ(graph.SlotCount(0), 4u);
    EXPECT_EQ(graph.SlotCount(1), 2u);
    EXPECT_EQ(graph.MemoryUsage(), sizeof(graph));
    EXPECT_THROW(HNSWGraphStorage(0, 2, 4), std::invalid_argument);
    EXPECT_THROW(HNSWGraphStorage(2, 0, 4), std::invalid_argument);
    EXPECT_THROW(HNSWGraphStorage(2, 2, 0), std::invalid_argument);
    EXPECT_THROW(HNSWGraphStorage(std::numeric_limits<size_t>::max(), 2, 4), std::length_error);
}

TEST(HNSWGraphStorage, AppendCopiesVectorAndInitializesAllLayersEmpty) {
    HNSWGraphStorage graph(2, 2, 4);
    std::array<float, 2> vector{3, 4};
    EXPECT_EQ(graph.Append(vector, 3), 0u);
    vector[0] = 99;
    EXPECT_EQ(graph.Vector(0)[0], 3);
    EXPECT_EQ(graph.Vector(0)[1], 4);
    EXPECT_EQ(graph.Level(0), 3u);
    for (uint8_t level = 0; level <= 3; ++level) {
        EXPECT_TRUE(graph.Neighbors(0, level).empty());
    }
    EXPECT_EQ(graph.EntryPoint(), 0u);
    EXPECT_EQ(graph.MaxLevel(), 3);
}

TEST(HNSWGraphStorage, EntryChangesOnlyWhenMaximumLayerIncreases) {
    HNSWGraphStorage graph(2, 2, 4);
    const std::array<float, 2> vector{0, 0};
    graph.Append(vector, 0);
    graph.Append(vector, 0);
    EXPECT_EQ(graph.EntryPoint(), 0u);
    graph.Append(vector, 2);
    EXPECT_EQ(graph.EntryPoint(), 2u);
    graph.Append(vector, 1);
    graph.Append(vector, 2);
    EXPECT_EQ(graph.EntryPoint(), 2u);
    EXPECT_EQ(graph.MaxLevel(), 2);
}

TEST(HNSWGraphStorage, BaseAndUpperRowsAreContiguousAndIndependent) {
    HNSWGraphStorage graph(2, 2, 4);
    const std::array<float, 2> vector{0, 0};
    graph.Append(vector, 2);
    graph.Append(vector, 0);
    graph.Append(vector, 1);
    EXPECT_EQ(graph.Neighbors(1, 0).data() - graph.Neighbors(0, 0).data(), 4);
    EXPECT_EQ(graph.Neighbors(2, 0).data() - graph.Neighbors(0, 0).data(), 8);
    EXPECT_EQ(graph.Neighbors(0, 2).data() - graph.Neighbors(0, 1).data(), 2);
    EXPECT_EQ(graph.Neighbors(2, 1).data() - graph.Neighbors(0, 1).data(), 4);
    const std::array<ann::idx_t, 2> base{1, 2};
    const std::array<ann::idx_t, 1> upper{2};
    graph.SetNeighbors(0, 0, base);
    graph.SetNeighbors(0, 1, upper);
    EXPECT_EQ(NeighborIds(graph, 0, 0), (std::vector<ann::idx_t>{1, 2}));
    EXPECT_EQ(NeighborIds(graph, 0, 1), (std::vector<ann::idx_t>{2}));
    EXPECT_TRUE(graph.Neighbors(0, 2).empty());
    // 存储层只写有向边，不偷偷补反向连接。
    EXPECT_TRUE(graph.Neighbors(2, 1).empty());
}

TEST(HNSWGraphStorage, FullRowsCanBeReplacedShortenedAndCleared) {
    HNSWGraphStorage graph(1, 2, 4);
    const std::array<float, 1> vector{0};
    for (int i = 0; i < 5; ++i) {
        graph.Append(vector, 1);
    }
    const std::array<ann::idx_t, 4> full{1, 2, 3, 4};
    graph.SetNeighbors(0, 0, full);
    EXPECT_EQ(graph.Neighbors(0, 0).size(), 4u);
    // 将邻接表自己的中段作为输入，覆盖与清尾不能破坏源数据。
    graph.SetNeighbors(0, 0, graph.Neighbors(0, 0).subspan(1, 2));
    EXPECT_EQ(NeighborIds(graph, 0, 0), (std::vector<ann::idx_t>{2, 3}));
    graph.SetNeighbors(0, 0, {});
    EXPECT_TRUE(graph.Neighbors(0, 0).empty());
}

TEST(HNSWGraphStorage, InvalidNeighborUpdatesPreserveOriginalRow) {
    HNSWGraphStorage graph(1, 1, 2);
    const std::array<float, 1> vector{0};
    graph.Append(vector, 1);
    graph.Append(vector, 1);
    graph.Append(vector, 0);
    const std::array<ann::idx_t, 1> valid{1};
    graph.SetNeighbors(0, 0, valid);
    const std::array<ann::idx_t, 1> self{0};
    const std::array<ann::idx_t, 1> invalid{ann::kInvalidIdx};
    const std::array<ann::idx_t, 2> duplicate{1, 1};
    const std::array<ann::idx_t, 3> too_many{1, 2, 1};
    const std::array<ann::idx_t, 1> wrong_level{2};
    EXPECT_THROW(graph.SetNeighbors(0, 0, self), std::invalid_argument);
    EXPECT_THROW(graph.SetNeighbors(0, 0, invalid), std::out_of_range);
    EXPECT_THROW(graph.SetNeighbors(0, 0, duplicate), std::invalid_argument);
    EXPECT_THROW(graph.SetNeighbors(0, 0, too_many), std::length_error);
    EXPECT_THROW(graph.SetNeighbors(0, 1, wrong_level), std::invalid_argument);
    EXPECT_EQ(NeighborIds(graph, 0, 0), (std::vector<ann::idx_t>{1}));
    EXPECT_TRUE(graph.Neighbors(0, 1).empty());
}

TEST(HNSWGraphStorage, InvalidNodeLayerAndDimensionAreRejected) {
    HNSWGraphStorage graph(2, 2, 4);
    const std::array<float, 1> wrong{0};
    const std::array<float, 2> vector{0, 1};
    EXPECT_THROW(graph.Vector(0), std::out_of_range);
    EXPECT_THROW(graph.Neighbors(0, 0), std::out_of_range);
    EXPECT_THROW(graph.Append(wrong, 0), std::invalid_argument);
    EXPECT_EQ(graph.Size(), 0u);
    graph.Append(vector, 0);
    EXPECT_THROW(graph.Level(ann::kInvalidIdx), std::out_of_range);
    EXPECT_THROW(graph.Neighbors(0, 1), std::out_of_range);
    EXPECT_THROW(graph.SetNeighbors(0, 1, {}), std::out_of_range);
    EXPECT_EQ(graph.Size(), 1u);
}

TEST(HNSWGraphStorage, ReallocationPreservesIdsVectorsAndUpperLinks) {
    HNSWGraphStorage graph(2, 2, 4);
    std::array<float, 2> vector{};
    for (size_t i = 0; i < 300; ++i) {
        vector = {static_cast<float>(i), -static_cast<float>(i)};
        const auto level = static_cast<uint8_t>(i % 3);
        EXPECT_EQ(graph.Append(vector, level), i);
        if (i == 2) {
            const std::array<ann::idx_t, 1> neighbor{1};
            graph.SetNeighbors(2, 1, neighbor);
        }
    }
    for (ann::idx_t i = 0; i < 300; ++i) {
        EXPECT_EQ(graph.Vector(i)[0], static_cast<float>(i));
        EXPECT_EQ(graph.Vector(i)[1], -static_cast<float>(i));
        EXPECT_EQ(graph.Level(i), i % 3);
    }
    EXPECT_EQ(NeighborIds(graph, 2, 1), (std::vector<ann::idx_t>{1}));
    EXPECT_EQ(graph.EntryPoint(), 2u);
}

TEST(HNSWGraphStorage, ArbitraryDimensionKeepsTightAlignedVectorStorage) {
    for (const size_t dim : {1u, 7u, 127u, 128u, 129u}) {
        HNSWGraphStorage graph(dim, 2, 4);
        const std::vector<float> vector(dim, 3.0f);
        graph.Append(vector, 0);
        graph.Append(vector, 0);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(graph.Vector(0).data()) % ann::kAlignment, 0u);
        EXPECT_EQ(graph.Vector(1).data() - graph.Vector(0).data(), static_cast<ptrdiff_t>(dim));
        EXPECT_EQ(graph.Vector(1).back(), 3.0f);
        EXPECT_GE(graph.MemoryUsage(),
                  sizeof(graph) + 2 * dim * sizeof(float) + 8 * sizeof(ann::idx_t) +
                      2 * sizeof(uint32_t) + 2);
    }
}

TEST(HNSWGraphStorage, AppendFromOwnVectorSurvivesGrowth) {
    HNSWGraphStorage graph(2, 2, 4);
    const std::array<float, 2> vector{7, 8};
    graph.Append(vector, 1);
    for (int i = 0; i < 20; ++i) {
        graph.Append(graph.Vector(0), 0);
    }
    EXPECT_EQ(graph.Vector(20)[0], 7);
    EXPECT_EQ(graph.Vector(20)[1], 8);
}

TEST(HNSWGraphStorage, UpperOffsetOverflowIsRejectedBeforeAllocation) {
    HNSWGraphStorage graph(2, ann::kInvalidIdx, 2);
    const std::array<float, 2> vector{0, 1};
    EXPECT_THROW(graph.Append(vector, 2), std::length_error);
    EXPECT_EQ(graph.Size(), 0u);
    EXPECT_EQ(graph.EntryPoint(), ann::kInvalidIdx);
    EXPECT_EQ(graph.MaxLevel(), -1);
    EXPECT_EQ(graph.MemoryUsage(), sizeof(graph));
    // 失败后仍可追加不需要上层池的节点。
    EXPECT_EQ(graph.Append(vector, 0), 0u);
}

TEST(HNSWGraphStorage, HighestUint8LayerHasIndependentSlots) {
    HNSWGraphStorage graph(1, 1, 2);
    const std::array<float, 1> vector{0};
    graph.Append(vector, 255);
    graph.Append(vector, 255);
    const std::array<ann::idx_t, 1> neighbor{1};
    graph.SetNeighbors(0, 255, neighbor);
    EXPECT_EQ(graph.MaxLevel(), 255);
    EXPECT_EQ(NeighborIds(graph, 0, 255), (std::vector<ann::idx_t>{1}));
    EXPECT_TRUE(graph.Neighbors(0, 254).empty());
}

TEST(HNSWGraphStorage, TenPointTwoDimensionalGraphRetainsLayerRelationships) {
    HNSWGraphStorage graph(2, 2, 4);
    const std::array<uint8_t, 10> levels{2, 0, 1, 0, 0, 2, 0, 1, 0, 0};
    for (size_t i = 0; i < levels.size(); ++i) {
        const std::array<float, 2> vector{static_cast<float>(i), static_cast<float>(i % 2)};
        graph.Append(vector, levels[i]);
    }
    for (ann::idx_t i = 0; i < 10; ++i) {
        std::array<ann::idx_t, 2> neighbors{};
        size_t count = 0;
        if (i > 0) {
            neighbors[count++] = i - 1;
        }
        if (i < 9) {
            neighbors[count++] = i + 1;
        }
        graph.SetNeighbors(i, 0, {neighbors.data(), count});
    }
    const std::array<ann::idx_t, 1> upper_forward{5};
    const std::array<ann::idx_t, 1> upper_reverse{0};
    graph.SetNeighbors(0, 2, upper_forward);
    graph.SetNeighbors(5, 2, upper_reverse);
    EXPECT_EQ(graph.Size(), 10u);
    EXPECT_EQ(NeighborIds(graph, 5, 0), (std::vector<ann::idx_t>{4, 6}));
    EXPECT_EQ(NeighborIds(graph, 0, 2), (std::vector<ann::idx_t>{5}));
    EXPECT_EQ(NeighborIds(graph, 5, 2), (std::vector<ann::idx_t>{0}));
    EXPECT_EQ(graph.EntryPoint(), 0u);
    EXPECT_EQ(graph.MaxLevel(), 2);
}

}  // namespace
