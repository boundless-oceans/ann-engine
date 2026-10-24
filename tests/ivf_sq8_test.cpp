#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ann/ivf.h"
#include "ann/kmeans.h"

namespace {

ann::IVFParams Params(size_t lists, size_t probes) {
    ann::IVFParams params;
    params.nlist = lists;
    params.nprobe = probes;
    params.kmeans_iters = 4;
    return params;
}

TEST(IVFSQ8, HandComputedResidualDistanceDoesNotAddCoarseDistance) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {8, 10, 12};
    index.Add(3, data);
    index.Build();
    const float query = 15;
    float distances[3]{};
    ann::idx_t ids[3]{};
    index.Search(&query, 3, {distances, ids, 3});
    EXPECT_STREQ(index.Name(), "IVFSQ8");
    EXPECT_EQ(ids[0], 2U);
    EXPECT_FLOAT_EQ(distances[0], 9);
    EXPECT_EQ(ids[1], 1U);
    EXPECT_NEAR(static_cast<double>(distances[1]), std::pow(5.0 - 2.0 / 255.0, 2), 0.00001);
    EXPECT_EQ(ids[2], 0U);
    EXPECT_FLOAT_EQ(distances[2], 49);
}

TEST(IVFSQ8, EachProbedListUsesItsOwnQueryResidual) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(2, 2), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {0, 2, 100, 102};
    index.Add(4, data);
    index.Build();
    const float query = 100;
    float distances[4]{};
    ann::idx_t ids[4]{};
    index.Search(&query, 4, {distances, ids, 4});
    const ann::idx_t expected[] = {2, 3, 1, 0};
    const float expected_distances[] = {0, 4, 9604, 10000};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ids[i], expected[i]);
        EXPECT_FLOAT_EQ(distances[i], expected_distances[i]);
    }
}

TEST(IVFSQ8, ADCMatchesSeparatelyDecodedResidualReference) {
    constexpr size_t dim = 3, count = 60, lists = 4, top_k = 7;
    std::mt19937 rng(1729);
    std::uniform_real_distribution<float> coordinate(-8, 8);
    std::vector<float> data(count * dim);
    for (auto& value : data) {
        value = coordinate(rng);
    }
    // 独立组装训练/解码参考，不访问索引私有中心、列表或代码。
    const auto params = Params(lists, lists);
    std::vector<float> centers(lists * dim), residuals(count * dim), decoded(count * dim);
    std::vector<uint32_t> assignments(count);
    ann::KMeans coarse(dim, lists, params.kmeans_iters, params.seed);
    coarse.Train(count, data.data(), centers.data());
    coarse.Assign(count, data.data(), centers.data(), assignments.data());
    for (size_t id = 0; id < count; ++id) {
        for (size_t d = 0; d < dim; ++d) {
            residuals[id * dim + d] = data[id * dim + d] - centers[assignments[id] * dim + d];
        }
    }
    ann::ScalarQuantizer reference(dim);
    reference.Train(count, residuals.data());
    uint8_t code[dim]{};
    for (size_t id = 0; id < count; ++id) {
        reference.Encode(residuals.data() + id * dim, code);
        reference.Decode(code, decoded.data() + id * dim);
    }
    ann::IVFIndex index(dim, ann::Metric::L2, params, std::make_unique<ann::ScalarQuantizer>(dim));
    index.Add(count, data.data());
    index.Build();
    for (size_t iteration = 0; iteration < 12; ++iteration) {
        float query[dim]{};
        for (auto& value : query) {
            value = coordinate(rng);
        }
        std::vector<std::pair<double, ann::idx_t>> expected;
        for (size_t id = 0; id < count; ++id) {
            double distance = 0;
            for (size_t d = 0; d < dim; ++d) {
                const float residual_query = query[d] - centers[assignments[id] * dim + d];
                const double delta = static_cast<double>(residual_query) -
                                     static_cast<double>(decoded[id * dim + d]);
                distance += delta * delta;
            }
            expected.emplace_back(distance, static_cast<ann::idx_t>(id));
        }
        std::sort(expected.begin(), expected.end());
        float distances[top_k]{};
        ann::idx_t ids[top_k]{};
        index.Search(query, top_k, {distances, ids, top_k});
        for (size_t i = 0; i < top_k; ++i) {
            EXPECT_EQ(ids[i], expected[i].second);
            EXPECT_NEAR(static_cast<double>(distances[i]),
                        expected[i].first,
                        std::max(1.0, expected[i].first) * 2e-7);
        }
    }
}

TEST(IVFSQ8, BuildReleasesRawStorageAndPreservesSize) {
    constexpr size_t count = 1024, dim = 8;
    ann::IVFIndex index(
        dim, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ScalarQuantizer>(dim));
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 37);
    }
    index.Add(count, data.data());
    const size_t before = index.MemoryUsage();
    index.Build();
    const size_t after = index.MemoryUsage();
    EXPECT_LT(after, before);
    EXPECT_EQ(index.Size(), count);
    EXPECT_GE(after,
              sizeof(index) + count * dim + count * sizeof(ann::idx_t) + dim * sizeof(float) +
                  2 * sizeof(uint32_t) + sizeof(ann::ScalarQuantizer) + 2 * dim * sizeof(float));
    index.Build();
    EXPECT_EQ(index.MemoryUsage(), after);
    EXPECT_EQ(index.Size(), count);
    EXPECT_THROW(index.Add(1, data.data()), std::logic_error);
    float distance{};
    ann::idx_t id{};
    EXPECT_NO_THROW(index.Search(data.data(), 1, {&distance, &id, 1}));
    EXPECT_EQ(index.MemoryUsage(), after);
}

TEST(IVFSQ8, BatchesCopyInputAndKeepOriginalIds) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ScalarQuantizer>(1));
    float first = 0;
    index.Add(1, &first);
    first = -999;
    const float second = 255;
    index.Add(1, &second);
    index.Build();
    float distance{};
    ann::idx_t id{};
    index.Search(&second, 1, {&distance, &id, 1});
    EXPECT_EQ(id, 1U);
    EXPECT_FLOAT_EQ(distance, 0);
    const float zero = 0;
    index.Search(&zero, 1, {&distance, &id, 1});
    EXPECT_EQ(id, 0U);
    EXPECT_FLOAT_EQ(distance, 0);
}

TEST(IVFSQ8, ProbeLimitAndSentinelBehaviorArePreserved) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(2, 1), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {0, 2, 100, 102};
    index.Add(4, data);
    index.Build();
    float distances[3]{};
    ann::idx_t ids[3]{};
    index.Search(data, 3, {distances, ids, 3});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 1U);
    EXPECT_EQ(ids[2], ann::kInvalidIdx);
    EXPECT_EQ(distances[2], std::numeric_limits<float>::max());
    index.SetNProbe(100);
    index.Search(data, 3, {distances, ids, 3});
    EXPECT_EQ(ids[2], 2U);
    index.SetNProbe(0);
    EXPECT_EQ(index.GetNProbe(), 1U);
}

TEST(IVFSQ8, EmptyListsAndConstantResidualsAreValid) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(3, 3), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {5, 5, 5, 5};
    index.Add(4, data);
    index.Build();
    const float query = 8;
    float distances[5]{};
    ann::idx_t ids[5]{};
    index.Search(&query, 5, {distances, ids, 5});
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ids[i], i);
        EXPECT_FLOAT_EQ(distances[i], 9);
    }
    EXPECT_EQ(ids[4], ann::kInvalidIdx);
}

TEST(IVFSQ8, IndependentQueriesDoNotShareLUTState) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(2, 2), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {0, 2, 100, 102};
    index.Add(4, data);
    index.Build();
    const auto& read_only = index;
    const float a = 0, b = 102;
    float distances_a[2]{}, distances_b[2]{}, repeated[2]{};
    ann::idx_t ids_a[2]{}, ids_b[2]{}, ids_repeated[2]{};
    const size_t memory = index.MemoryUsage();
    read_only.Search(&a, 2, {distances_a, ids_a, 2});
    read_only.Search(&b, 2, {distances_b, ids_b, 2});
    read_only.Search(&a, 2, {repeated, ids_repeated, 2});
    EXPECT_EQ(ids_a[0], 0U);
    EXPECT_EQ(ids_b[0], 3U);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(ids_a[i], ids_repeated[i]);
        EXPECT_FLOAT_EQ(distances_a[i], repeated[i]);
    }
    EXPECT_EQ(index.MemoryUsage(), memory);
}

TEST(IVFSQ8, InvalidQueryPreservesOutputAndCanBeFollowedByValidQuery) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {0, 1};
    index.Add(2, data);
    index.Build();
    const float huge = std::numeric_limits<float>::max();
    float distance = 123;
    ann::idx_t id = 123;
    EXPECT_THROW(index.Search(&huge, 1, {&distance, &id, 1}), std::overflow_error);
    EXPECT_FLOAT_EQ(distance, 123);
    EXPECT_EQ(id, 123U);
    index.Search(data, 1, {&distance, &id, 1});
    EXPECT_EQ(id, 0U);
}

TEST(IVFSQ8, ResidualOverflowDoesNotPublishIndexAndBuildCanBeRetried) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ScalarQuantizer>(1));
    const float limit = std::numeric_limits<float>::max();
    const float data[] = {-limit, limit, limit};
    index.Add(3, data);
    EXPECT_THROW(index.Build(), std::overflow_error);
    EXPECT_EQ(index.Size(), 3U);
    float distance{};
    ann::idx_t id{};
    EXPECT_THROW(index.Search(data, 1, {&distance, &id, 1}), std::logic_error);
    const float extra = -limit;
    EXPECT_NO_THROW(index.Add(1, &extra));
    EXPECT_NO_THROW(index.Build());
    EXPECT_EQ(index.Size(), 4U);
}

TEST(IVFSQ8, RejectsMismatchedQuantizerDimensions) {
    EXPECT_THROW(
        ann::IVFIndex(1, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ScalarQuantizer>(2)),
        std::invalid_argument);
    EXPECT_THROW(
        ann::IVFIndex(2, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ScalarQuantizer>(1)),
        std::invalid_argument);
}

TEST(IVFSQ8, MissingSamplesCanBeAddedBeforeRetry) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(2, 2), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {1, 10};
    index.Add(1, data);
    EXPECT_THROW(index.Build(), std::invalid_argument);
    index.Add(1, data + 1);
    index.Build();
    float distance{};
    ann::idx_t id{};
    index.Search(data + 1, 1, {&distance, &id, 1});
    EXPECT_EQ(id, 1U);
    EXPECT_FLOAT_EQ(distance, 0);
}

TEST(IVFSQ8, CapacityAndRequestedCountBoundWrites) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {0, 255};
    index.Add(2, data);
    index.Build();
    float distances[] = {-1, -1};
    ann::idx_t ids[] = {99, 99};
    index.Search(data, 2, {distances, ids, 1});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 99U);
    index.Search(data, 1, {distances, ids, 2});
    index.Search(data, 0, {distances, ids, 2});
    index.Search(nullptr, 2, {distances, ids, 2});
    EXPECT_EQ(ids[1], 99U);
    EXPECT_FLOAT_EQ(distances[1], -1);
}

TEST(IVFSQ8, BatchQueriesUseSeparateOutputRows) {
    ann::IVFIndex index(
        1, ann::Metric::L2, Params(2, 2), std::make_unique<ann::ScalarQuantizer>(1));
    const float data[] = {0, 2, 100, 102};
    const float queries[] = {0, 102};
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

}  // namespace
