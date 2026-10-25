#include "ann/ivf.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ann/flat.h"

namespace {

ann::IVFParams Params(size_t nlist, size_t nprobe) {
    ann::IVFParams params;
    params.nlist = nlist;
    params.nprobe = nprobe;
    params.kmeans_iters = 4;
    return params;
}

TEST(IVF, HandComputedSingleListAndOriginalIds) {
    ann::IVFIndex index(2, ann::Metric::L2, Params(1, 1));
    const float data[] = {3, 4, 0, 0, 1, 0};
    index.Add(3, data);
    index.Build();
    const float query[] = {0, 0};
    float distances[3]{};
    ann::idx_t ids[3]{};
    index.Search(query, 3, {distances, ids, 3});
    EXPECT_EQ(ids[0], 1U);
    EXPECT_EQ(ids[1], 2U);
    EXPECT_EQ(ids[2], 0U);
    EXPECT_FLOAT_EQ(distances[0], 0);
    EXPECT_FLOAT_EQ(distances[1], 1);
    EXPECT_FLOAT_EQ(distances[2], 25);
    EXPECT_EQ(index.Size(), 3U);
    EXPECT_EQ(index.Dim(), 2U);
    EXPECT_EQ(index.GetMetric(), ann::Metric::L2);
    EXPECT_STREQ(index.Name(), "IVFFlat");
}

TEST(IVF, AddCopiesInputAndPreservesIdsAcrossBatches) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(2, 2));
    float first[] = {20, 10};
    index.Add(2, first);
    first[0] = -999;
    const float second[] = {1, 0};
    index.Add(2, second);
    index.Build();
    const float query = 0;
    float distances[4]{};
    ann::idx_t ids[4]{};
    index.Search(&query, 4, {distances, ids, 4});
    EXPECT_EQ(ids[0], 3U);
    EXPECT_EQ(ids[1], 2U);
    EXPECT_EQ(ids[2], 1U);
    EXPECT_EQ(ids[3], 0U);
    EXPECT_FLOAT_EQ(distances[3], 400);
}

TEST(IVF, BuildFailureCanBeRetriedAfterMoreData) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(3, 3));
    EXPECT_THROW(index.Build(), std::invalid_argument);
    const float data[] = {0, 1, 10};
    index.Add(2, data);
    const size_t memory = index.MemoryUsage();
    EXPECT_THROW(index.Build(), std::invalid_argument);
    EXPECT_EQ(index.Size(), 2U);
    EXPECT_EQ(index.MemoryUsage(), memory);
    index.Add(1, data + 2);
    EXPECT_NO_THROW(index.Build());
    EXPECT_EQ(index.Size(), 3U);
}

TEST(IVF, BuiltIndexRejectsAddAndRepeatedBuildIsNoOp) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(1, 1));
    const float point = 2;
    index.Add(1, &point);
    index.Build();
    const size_t memory = index.MemoryUsage();
    EXPECT_THROW(index.Add(1, &point), std::logic_error);
    EXPECT_THROW(index.Add(0, nullptr), std::logic_error);
    EXPECT_NO_THROW(index.Build());
    EXPECT_EQ(index.Size(), 1U);
    EXPECT_EQ(index.MemoryUsage(), memory);
}

TEST(IVF, SearchBeforeBuildRejectsRealQueryButAllowsNoOp) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(1, 1));
    const float query = 0;
    float distance = 123;
    ann::idx_t id = 123;
    EXPECT_THROW(index.Search(&query, 1, {&distance, &id, 1}), std::logic_error);
    EXPECT_FLOAT_EQ(distance, 123);
    EXPECT_EQ(id, 123U);
    EXPECT_NO_THROW(index.Search(nullptr, 0, {}));
}

TEST(IVF, NProbeRestrictsScanningAndFillsMissingResults) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(2, 1));
    const float data[] = {0, 1, 100, 101};
    index.Add(4, data);
    index.Build();
    const float query = 0;
    float distances[3]{};
    ann::idx_t ids[3]{};
    index.Search(&query, 3, {distances, ids, 3});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 1U);
    EXPECT_EQ(ids[2], ann::kInvalidIdx);
    EXPECT_EQ(distances[2], std::numeric_limits<float>::max());
    index.SetNProbe(2);
    index.Search(&query, 3, {distances, ids, 3});
    EXPECT_EQ(ids[2], 2U);
    EXPECT_FLOAT_EQ(distances[2], 10000);
}

TEST(IVF, AllListsMatchFlatAndIndependentDoubleReference) {
    constexpr size_t dim = 3, count = 90, top_k = 7;
    std::mt19937 rng(923);
    std::uniform_real_distribution<float> coordinate(-10, 10);
    std::vector<float> data(count * dim);
    for (auto& value : data) {
        value = coordinate(rng);
    }
    ann::IVFIndex index(dim, ann::Metric::L2, Params(7, 7));
    ann::FlatIndex flat(dim, ann::Metric::L2);
    index.Add(count, data.data());
    flat.Add(count, data.data());
    index.Build();
    const size_t memory = index.MemoryUsage();
    for (size_t repetition = 0; repetition < 20; ++repetition) {
        float query[dim];
        for (auto& value : query) {
            value = coordinate(rng);
        }
        float actual[top_k]{}, expected[top_k]{};
        ann::idx_t ids[top_k]{}, flat_ids[top_k]{};
        index.Search(query, top_k, {actual, ids, top_k});
        flat.Search(query, top_k, {expected, flat_ids, top_k});
        std::vector<std::pair<double, ann::idx_t>> reference;
        for (size_t i = 0; i < count; ++i) {
            double distance = 0;
            for (size_t d = 0; d < dim; ++d) {
                const double delta =
                    static_cast<double>(query[d]) - static_cast<double>(data[i * dim + d]);
                distance += delta * delta;
            }
            reference.emplace_back(distance, static_cast<ann::idx_t>(i));
        }
        std::sort(reference.begin(), reference.end());
        for (size_t k = 0; k < top_k; ++k) {
            EXPECT_EQ(ids[k], flat_ids[k]);
            EXPECT_EQ(ids[k], reference[k].second);
            EXPECT_FLOAT_EQ(actual[k], expected[k]);
            EXPECT_NEAR(static_cast<double>(actual[k]), reference[k].first, 0.0001);
        }
    }
    EXPECT_EQ(index.MemoryUsage(), memory);
    EXPECT_EQ(index.GetNProbe(), 7U);
    EXPECT_EQ(index.Size(), count);
}

TEST(IVF, DuplicatePointsAndEmptyListsKeepDistinctIds) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(3, 3));
    const float data[] = {5, 5, 5, 5};
    index.Add(4, data);
    index.Build();
    float distances[5]{};
    ann::idx_t ids[5]{};
    const float query = 5;
    index.Search(&query, 5, {distances, ids, 5});
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ids[i], i);
        EXPECT_FLOAT_EQ(distances[i], 0);
    }
    EXPECT_EQ(ids[4], ann::kInvalidIdx);
}

TEST(IVF, EqualDistancesUseLowerOriginalIdAcrossLists) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(2, 2));
    const float data[] = {1, -1, 3, -3};
    const float query = 0;
    index.Add(4, data);
    index.Build();
    float distances[3]{};
    ann::idx_t ids[3]{};
    index.Search(&query, 3, {distances, ids, 3});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 1U);
    EXPECT_EQ(ids[2], 2U);
}

TEST(IVF, OutputCapacityAndRequestedCountBoundWrites) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(1, 1));
    const float data[] = {0, 1, 2};
    index.Add(3, data);
    index.Build();
    float distances[] = {-1, -1, -1};
    ann::idx_t ids[] = {99, 99, 99};
    index.Search(data, 3, {distances, ids, 1});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 99U);
    index.Search(data, 1, {distances, ids, 3});
    EXPECT_EQ(ids[1], 99U);
    EXPECT_FLOAT_EQ(distances[1], -1);
    index.Search(data, 0, {distances, ids, 3});
    index.Search(nullptr, 3, {distances, ids, 3});
    index.Search(data, 3, {nullptr, ids, 3});
    index.Search(data, 3, {distances, nullptr, 3});
    EXPECT_EQ(ids[2], 99U);
}

TEST(IVF, BatchQueriesUseIndependentOutputRows) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(2, 2));
    const float data[] = {0, 1, 10, 11};
    const float queries[] = {0, 11};
    index.Add(4, data);
    index.Build();
    float distances[4]{};
    ann::idx_t ids[4]{};
    index.SearchBatch(queries, 2, 2, {distances, ids, 2});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 1U);
    EXPECT_EQ(ids[2], 3U);
    EXPECT_EQ(ids[3], 2U);
}

TEST(IVF, ProbeSetterClampsZeroAndQueryClampsLargeValue) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(1, 1));
    const float data[] = {0, 1};
    index.Add(2, data);
    index.Build();
    index.SetNProbe(0);
    EXPECT_EQ(index.GetNProbe(), 1U);
    index.SetNProbe(std::numeric_limits<size_t>::max());
    EXPECT_EQ(index.GetNProbe(), std::numeric_limits<size_t>::max());
    float distances[2]{};
    ann::idx_t ids[2]{};
    index.Search(data, 2, {distances, ids, 2});
    EXPECT_EQ(ids[1], 1U);
}

TEST(IVF, InvalidAddAndQueryPreserveStateAndOutput) {
    ann::IVFIndex index(1, ann::Metric::L2, Params(1, 1));
    const float data[] = {0, 1};
    index.Add(2, data);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(index.Add(1, &nan), std::invalid_argument);
    EXPECT_THROW(index.Add(1, nullptr), std::invalid_argument);
    EXPECT_THROW(index.Add(std::numeric_limits<size_t>::max(), data), std::length_error);
    EXPECT_NO_THROW(index.Add(0, nullptr));
    EXPECT_EQ(index.Size(), 2U);
    index.Build();
    float distance = 123;
    ann::idx_t id = 123;
    EXPECT_THROW(index.Search(&nan, 1, {&distance, &id, 1}), std::invalid_argument);
    const float extreme = std::numeric_limits<float>::max();
    EXPECT_THROW(index.Search(&extreme, 1, {&distance, &id, 1}), std::invalid_argument);
    EXPECT_EQ(id, 123U);
    EXPECT_FLOAT_EQ(distance, 123);
    EXPECT_NO_THROW(index.Search(data, 1, {&distance, &id, 1}));
    EXPECT_EQ(id, 0U);
}

TEST(IVF, RejectsInvalidConfigurationAndUnsupportedMetrics) {
    EXPECT_THROW(ann::IVFIndex(0, ann::Metric::L2), std::invalid_argument);
    EXPECT_THROW(ann::IVFIndex(1, ann::Metric::InnerProduct), std::invalid_argument);
    EXPECT_THROW(ann::IVFIndex(1, ann::Metric::Cosine), std::invalid_argument);
    EXPECT_THROW(ann::IVFIndex(1, static_cast<ann::Metric>(99)), std::invalid_argument);
    EXPECT_THROW(ann::IVFIndex(1, ann::Metric::L2, Params(0, 1)), std::invalid_argument);
    EXPECT_THROW(ann::IVFIndex(1, ann::Metric::L2, Params(1, 0)), std::invalid_argument);
    auto params = Params(1, 1);
    params.kmeans_iters = 0;
    EXPECT_THROW(ann::IVFIndex(1, ann::Metric::L2, params), std::invalid_argument);
    EXPECT_THROW(ann::IVFIndex(std::numeric_limits<size_t>::max(), ann::Metric::L2, Params(2, 1)),
                 std::length_error);
}

TEST(IVF, MemoryUsageIncludesOwnedArraysAndObject) {
    ann::IVFIndex index(2, ann::Metric::L2, Params(2, 2));
    const size_t empty = index.MemoryUsage();
    EXPECT_GE(empty, sizeof(index));
    const float data[] = {0, 0, 1, 1, 9, 9, 10, 10};
    index.Add(4, data);
    const size_t before = index.MemoryUsage();
    EXPECT_GE(before, empty + sizeof(data));
    index.Build();
    // 两个二维中心、三个偏移和四个成员 ID；工作区不计入持久存储。
    EXPECT_GE(index.MemoryUsage(), before + 4 * sizeof(float) + 7 * sizeof(uint32_t));
}

class TestQuantizer final : public ann::Quantizer {
public:
    size_t Dim() const override { return 1; }
    void Train(size_t, const float*) override {}
    size_t CodeSize() const override { return 1; }
    void Encode(const float*, uint8_t*) const override {}
    void Decode(const uint8_t*, float*) const override {}
    void BuildLUT(const float*, float*) const override {}
    size_t LutSize() const override { return 1; }
    float DistanceFromLUT(const float*, const uint8_t*) const override { return 0; }
    size_t MemoryUsage() const override { return 0; }
    const char* Name() const override { return "Test"; }
};

TEST(IVF, RejectsUnsupportedQuantizerTypes) {
    EXPECT_THROW(ann::IVFIndex(1, ann::Metric::L2, Params(1, 1), std::make_unique<TestQuantizer>()),
                 std::invalid_argument);
}

}  // namespace
