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

const float kGrid[] = {0, 0, 10, 10, 0, 0, 12, 12, 2, 2, 10, 10, 2, 2, 12, 12};

TEST(IVFPQ, BaseDimensionIsIndependentOfCodeSize) {
    const ann::ProductQuantizer pq(8, 2, 4);
    const ann::ScalarQuantizer sq(8);
    const ann::Quantizer& base_pq = pq;
    const ann::Quantizer& base_sq = sq;
    EXPECT_EQ(base_pq.Dim(), 8U);
    EXPECT_EQ(base_pq.CodeSize(), 2U);
    EXPECT_EQ(base_sq.Dim(), 8U);
    EXPECT_EQ(base_sq.CodeSize(), 8U);
    EXPECT_NO_THROW(ann::IVFIndex(
        8, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(8, 2, 4)));
}

TEST(IVFPQ, HandComputedResidualDistancesAndShortCodeStride) {
    ann::IVFIndex index(
        4, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(4, 2, 2));
    index.Add(4, kGrid);
    index.Build();
    const float query[] = {3, 3, 13, 13};
    float distances[4]{};
    ann::idx_t ids[4]{};
    index.Search(query, 4, {distances, ids, 4});
    const ann::idx_t expected[] = {3, 1, 2, 0};
    const float expected_distances[] = {4, 20, 20, 36};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ids[i], expected[i]);
        EXPECT_FLOAT_EQ(distances[i], expected_distances[i]);
    }
    EXPECT_STREQ(index.Name(), "IVFPQ");
    EXPECT_EQ(index.Size(), 4U);
}

TEST(IVFPQ, SeparateListsUseSeparateQueryResiduals) {
    ann::IVFIndex index(
        2, ann::Metric::L2, Params(2, 2), std::make_unique<ann::ProductQuantizer>(2, 1, 2));
    const float data[] = {0, 0, 2, 2, 100, 100, 102, 102};
    index.Add(4, data);
    index.Build();
    const float query[] = {100, 100};
    float distances[4]{};
    ann::idx_t ids[4]{};
    index.Search(query, 4, {distances, ids, 4});
    const ann::idx_t expected[] = {2, 3, 1, 0};
    const float expected_distances[] = {0, 8, 19208, 20000};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ids[i], expected[i]);
        EXPECT_FLOAT_EQ(distances[i], expected_distances[i]);
    }
}

TEST(IVFPQ, ADCMatchesSeparatelyDecodedResidualReference) {
    constexpr size_t dim = 8, segments = 2, count = 60, lists = 4, top_k = 7;
    std::mt19937 rng(1297);
    std::uniform_real_distribution<float> coordinate(-8, 8);
    std::vector<float> data(count * dim);
    for (auto& value : data) {
        value = coordinate(rng);
    }
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
    ann::ProductQuantizer reference(dim, segments, 8, 17);
    reference.Train(count, residuals.data());
    uint8_t code[segments]{};
    for (size_t id = 0; id < count; ++id) {
        reference.Encode(residuals.data() + id * dim, code);
        reference.Decode(code, decoded.data() + id * dim);
    }
    ann::IVFIndex index(dim,
                        ann::Metric::L2,
                        params,
                        std::make_unique<ann::ProductQuantizer>(dim, segments, 8, 17));
    index.Add(count, data.data());
    index.Build();
    for (size_t repeat = 0; repeat < 12; ++repeat) {
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

TEST(IVFPQ, Default256CodebookBuildsAndQueries) {
    constexpr size_t count = 256, dim = 4;
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        data[i * dim] = static_cast<float>(i);
        data[i * dim + 1] = static_cast<float>(i + 1);
        data[i * dim + 2] = static_cast<float>(1000 + i);
        data[i * dim + 3] = static_cast<float>(3000 + i);
    }
    ann::IVFIndex index(
        dim, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(dim, 2));
    index.Add(count, data.data());
    index.Build();
    float distance{};
    ann::idx_t id{};
    index.Search(data.data() + 173 * dim, 1, {&distance, &id, 1});
    EXPECT_EQ(id, 173U);
    EXPECT_FLOAT_EQ(distance, 0);
}

TEST(IVFPQ, BuildReleasesRawStorageAndCountsActualCodeLength) {
    constexpr size_t count = 1024, dim = 16, segments = 2;
    ann::IVFIndex index(dim,
                        ann::Metric::L2,
                        Params(1, 1),
                        std::make_unique<ann::ProductQuantizer>(dim, segments, 2));
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 17);
    }
    index.Add(count, data.data());
    const size_t before = index.MemoryUsage();
    index.Build();
    const size_t after = index.MemoryUsage();
    EXPECT_LT(after, before / 4);
    EXPECT_GE(after,
              sizeof(index) + count * segments + count * sizeof(ann::idx_t) + dim * sizeof(float) +
                  2 * sizeof(uint32_t) + sizeof(ann::ProductQuantizer) + 2 * dim * sizeof(float));
    EXPECT_EQ(index.Size(), count);
    index.Build();
    EXPECT_EQ(index.MemoryUsage(), after);
    EXPECT_THROW(index.Add(1, data.data()), std::logic_error);
}

TEST(IVFPQ, MismatchedDimensionsCannotBeHiddenByMatchingCodeSize) {
    EXPECT_THROW(
        ann::IVFIndex(
            2, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(4, 2, 2)),
        std::invalid_argument);
    EXPECT_THROW(
        ann::IVFIndex(
            4, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(2, 2, 2)),
        std::invalid_argument);
}

TEST(IVFPQ, InsufficientPQTrainingDataCanBeAppendedBeforeRetry) {
    ann::IVFIndex index(
        4, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(4, 2, 4));
    index.Add(3, kGrid);
    EXPECT_THROW(index.Build(), std::invalid_argument);
    EXPECT_EQ(index.Size(), 3U);
    float distances[1]{};
    ann::idx_t ids[1]{};
    EXPECT_THROW(index.Search(kGrid, 1, {distances, ids, 1}), std::logic_error);
    index.Add(1, kGrid + 12);
    EXPECT_NO_THROW(index.Build());
    EXPECT_EQ(index.Size(), 4U);
    EXPECT_NO_THROW(index.Search(kGrid, 1, {distances, ids, 1}));
    EXPECT_EQ(ids[0], 0U);
}

TEST(IVFPQ, ProbeLimitsAndMissingResultsRemainExplicit) {
    ann::IVFIndex index(
        2, ann::Metric::L2, Params(2, 1), std::make_unique<ann::ProductQuantizer>(2, 1, 2));
    const float data[] = {0, 0, 2, 2, 100, 100, 102, 102};
    const float query[] = {100, 100};
    index.Add(4, data);
    index.Build();
    float distances[3]{};
    ann::idx_t ids[3]{};
    index.Search(query, 3, {distances, ids, 3});
    EXPECT_EQ(ids[0], 2U);
    EXPECT_EQ(ids[1], 3U);
    EXPECT_EQ(ids[2], ann::kInvalidIdx);
    EXPECT_EQ(distances[2], std::numeric_limits<float>::max());
    index.SetNProbe(std::numeric_limits<size_t>::max());
    index.Search(query, 3, {distances, ids, 3});
    EXPECT_EQ(ids[2], 1U);
    index.SetNProbe(0);
    EXPECT_EQ(index.GetNProbe(), 1U);
}

TEST(IVFPQ, EmptyListsAndDuplicateCentersKeepDistinctOriginalIds) {
    ann::IVFIndex index(
        4, ann::Metric::L2, Params(3, 3), std::make_unique<ann::ProductQuantizer>(4, 2, 4));
    const float data[] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    index.Add(4, data);
    index.Build();
    const float query[] = {6, 6, 6, 6};
    float distances[5]{};
    ann::idx_t ids[5]{};
    index.Search(query, 5, {distances, ids, 5});
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ids[i], i);
        EXPECT_FLOAT_EQ(distances[i], 4);
    }
    EXPECT_EQ(ids[4], ann::kInvalidIdx);
}

TEST(IVFPQ, SingleCenterQuantizationAndBatchOutputRows) {
    ann::IVFIndex single(
        2, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(2, 1, 1));
    const float data[] = {-2, -4, 2, 4};
    single.Add(2, data);
    single.Build();
    const float query[] = {1, 2};
    float distances[2]{};
    ann::idx_t ids[2]{};
    single.Search(query, 2, {distances, ids, 2});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 1U);
    EXPECT_FLOAT_EQ(distances[0], 5);
    EXPECT_FLOAT_EQ(distances[1], 5);

    ann::IVFIndex index(
        4, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(4, 2, 2));
    index.Add(4, kGrid);
    index.Build();
    float queries[8]{};
    std::copy_n(kGrid, 4, queries);
    std::copy_n(kGrid + 12, 4, queries + 4);
    index.SearchBatch(queries, 2, 1, {distances, ids, 1});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 3U);
}

TEST(IVFPQ, IndependentQueriesAndInvalidQueryDoNotCorruptState) {
    ann::IVFIndex index(
        4, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(4, 2, 2));
    index.Add(4, kGrid);
    index.Build();
    const auto& read_only = index;
    const size_t memory = index.MemoryUsage();
    float distance = 123;
    ann::idx_t id = 123;
    const float huge[] = {std::numeric_limits<float>::max(), 0, 0, 0};
    EXPECT_THROW(read_only.Search(huge, 1, {&distance, &id, 1}), std::overflow_error);
    EXPECT_EQ(id, 123U);
    EXPECT_FLOAT_EQ(distance, 123);
    read_only.Search(kGrid, 1, {&distance, &id, 1});
    EXPECT_EQ(id, 0U);
    read_only.Search(kGrid + 12, 1, {&distance, &id, 1});
    EXPECT_EQ(id, 3U);
    read_only.Search(kGrid, 1, {&distance, &id, 1});
    EXPECT_EQ(id, 0U);
    EXPECT_EQ(index.MemoryUsage(), memory);
    EXPECT_EQ(index.Size(), 4U);
}

TEST(IVFPQ, OutputCapacityBoundsAndCopiedBatchesKeepIds) {
    ann::IVFIndex index(
        4, ann::Metric::L2, Params(1, 1), std::make_unique<ann::ProductQuantizer>(4, 2, 2));
    float first[8]{};
    std::copy_n(kGrid, 8, first);
    index.Add(2, first);
    std::fill(std::begin(first), std::end(first), -999);
    index.Add(2, kGrid + 8);
    index.Build();
    float distances[] = {-7, -7};
    ann::idx_t ids[] = {99, 99};
    index.Search(kGrid, 2, {distances, ids, 1});
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 99U);
    index.Search(kGrid, 1, {distances, ids, 2});
    index.Search(kGrid, 0, {distances, ids, 2});
    index.Search(nullptr, 2, {distances, ids, 2});
    EXPECT_EQ(ids[1], 99U);
    EXPECT_FLOAT_EQ(distances[1], -7);
}

}  // namespace
