#include "ann/hnsw.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "ann/flat.h"

namespace {

std::vector<float> UnitVectors(size_t count, size_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> vectors(count * dim);
    for (size_t i = 0; i < count; ++i) {
        float squared_norm = 0;
        for (size_t d = 0; d < dim; ++d) {
            auto& value = vectors[i * dim + d];
            value = distribution(rng);
            squared_norm += value * value;
        }
        const float norm = std::sqrt(squared_norm);
        for (size_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] /= norm;
        }
    }
    return vectors;
}

TEST(HNSWIndex, EmptyIndexMetadataAndSentinels) {
    const ann::HNSWIndex index(2, ann::Metric::L2);
    EXPECT_EQ(index.Size(), 0u);
    EXPECT_EQ(index.Dim(), 2u);
    EXPECT_EQ(index.GetMetric(), ann::Metric::L2);
    EXPECT_STREQ(index.Name(), "HNSW");
    EXPECT_EQ(index.GetStats().max_level, -1);
    EXPECT_EQ(index.GetStats().nodes, 0u);
    const float query[2] = {0, 0};
    float distances[2]{};
    ann::idx_t ids[2]{};
    index.Search(query, 2, {distances, ids, 2});
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(ids[i], ann::kInvalidIdx);
        EXPECT_EQ(distances[i], std::numeric_limits<float>::max());
    }
}

TEST(HNSWIndex, SmallUnprunedGraphMatchesHandComputedL2) {
    const float base[] = {0, 2, 5, 9};
    ann::HNSWIndex index(1, ann::Metric::L2);
    index.Add(4, base);
    const float query[] = {4};
    float distances[4]{};
    ann::idx_t ids[4]{};
    index.Search(query, 4, {distances, ids, 4});
    EXPECT_EQ((std::vector<ann::idx_t>{ids, ids + 4}), (std::vector<ann::idx_t>{2, 1, 0, 3}));
    EXPECT_FLOAT_EQ(distances[0], 1);
    EXPECT_FLOAT_EQ(distances[1], 4);
    EXPECT_FLOAT_EQ(distances[2], 16);
    EXPECT_FLOAT_EQ(distances[3], 25);
}

TEST(HNSWIndex, SmallGraphOrdersNegativeInnerProduct) {
    const float base[] = {-2, 1, 3, 8};
    ann::HNSWIndex index(1, ann::Metric::InnerProduct);
    index.Add(4, base);
    const float query[] = {2};
    float distances[2]{};
    ann::idx_t ids[2]{};
    index.Search(query, 2, {distances, ids, 2});
    EXPECT_EQ(ids[0], 3u);
    EXPECT_EQ(ids[1], 2u);
    EXPECT_FLOAT_EQ(distances[0], -16);
    EXPECT_FLOAT_EQ(distances[1], -6);
}

TEST(HNSWIndex, OutputBoundsZeroRequestsAndNullPointers) {
    const float base[] = {0, 1, 2};
    ann::HNSWIndex index(1, ann::Metric::L2);
    index.Add(3, base);
    const float query[] = {0};
    float distances[4] = {-99, -99, -99, -99};
    ann::idx_t ids[4] = {99, 99, 99, 99};
    index.Search(query, 0, {distances, ids, 4});
    EXPECT_EQ(ids[0], 99u);
    index.Search(nullptr, 3, {distances, ids, 4});
    index.Search(query, 3, {nullptr, ids, 4});
    index.Search(query, 3, {distances, nullptr, 4});
    EXPECT_EQ(ids[0], 99u);
    index.Search(query, 3, {distances, ids, 1});
    EXPECT_EQ(ids[0], 0u);
    EXPECT_EQ(ids[1], 99u);
    index.Search(query, 1, {distances, ids, 4});
    EXPECT_EQ(ids[1], 99u);
}

TEST(HNSWIndex, TooFewResultsFillOnlyRequestedSentinels) {
    const float base[] = {1, 2};
    ann::HNSWIndex index(1, ann::Metric::L2);
    index.Add(2, base);
    const float query[] = {0};
    float distances[5] = {-1, -1, -1, -1, -1};
    ann::idx_t ids[5] = {99, 99, 99, 99, 99};
    index.Search(query, 4, {distances, ids, 5});
    EXPECT_EQ(ids[0], 0u);
    EXPECT_EQ(ids[1], 1u);
    EXPECT_EQ(ids[2], ann::kInvalidIdx);
    EXPECT_EQ(ids[3], ann::kInvalidIdx);
    EXPECT_EQ(ids[4], 99u);
    EXPECT_EQ(distances[3], std::numeric_limits<float>::max());
}

TEST(HNSWIndex, MultipleAddsCopyInputAndKeepStableIds) {
    ann::HNSWIndex index(1, ann::Metric::L2);
    float first[] = {0, 2};
    const float second[] = {5, 9};
    index.Add(2, first);
    first[0] = 100;
    index.Add(0, nullptr);
    index.Add(2, nullptr);
    index.Add(2, second);
    const float query[] = {0};
    float distance = -1;
    ann::idx_t id = ann::kInvalidIdx;
    index.Search(query, 1, {&distance, &id, 1});
    EXPECT_EQ(index.Size(), 4u);
    EXPECT_EQ(id, 0u);
    EXPECT_FLOAT_EQ(distance, 0);
}

TEST(HNSWIndex, RuntimeEfIsClampedAndQueryWidthCoversK) {
    ann::HNSWIndex index(1, ann::Metric::L2);
    const auto initial_ef = index.Params().ef_search;
    index.SetEfSearch(0);
    EXPECT_EQ(index.GetEfSearch(), 1u);
    EXPECT_EQ(index.Params().ef_search, initial_ef);
    const float base[] = {0, 1, 2, 3};
    index.Add(4, base);
    const float query[] = {0};
    float distances[4]{};
    ann::idx_t ids[4]{};
    index.Search(query, 4, {distances, ids, 4});
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ids[i], i);
    }
}

TEST(HNSWIndex, ParameterValidation) {
    EXPECT_THROW(ann::HNSWIndex(0, ann::Metric::L2), std::invalid_argument);
    EXPECT_THROW(ann::HNSWIndex(1, static_cast<ann::Metric>(255)), std::invalid_argument);
    ann::HNSWParams params;
    params.M = 1;
    EXPECT_THROW(ann::HNSWIndex(1, ann::Metric::L2, params), std::invalid_argument);
    params = {};
    params.Mmax0 = 1;
    EXPECT_THROW(ann::HNSWIndex(1, ann::Metric::L2, params), std::invalid_argument);
    params = {};
    params.ef_construction = 0;
    EXPECT_THROW(ann::HNSWIndex(1, ann::Metric::L2, params), std::invalid_argument);
    params = {};
    params.ef_search = 0;
    EXPECT_THROW(ann::HNSWIndex(1, ann::Metric::L2, params), std::invalid_argument);
}

TEST(HNSWIndex, InvalidBatchAndOverflowDoNotAppendNodes) {
    ann::HNSWIndex index(1, ann::Metric::L2);
    const float good[] = {0};
    index.Add(1, good);
    const float invalid[] = {1, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(index.Add(2, invalid), std::invalid_argument);
    EXPECT_THROW(index.Add(std::numeric_limits<size_t>::max(), good), std::length_error);
    EXPECT_EQ(index.Size(), 1u);
}

TEST(HNSWIndex, FailedDistancePlanningPreservesGraphAndRandomSequence) {
    ann::HNSWIndex tested(1, ann::Metric::L2);
    ann::HNSWIndex reference(1, ann::Metric::L2);
    const float first[] = {0};
    const float overflow[] = {std::numeric_limits<float>::max()};
    const float next[] = {1};
    tested.Add(1, first);
    reference.Add(1, first);
    EXPECT_THROW(tested.Add(1, overflow), std::invalid_argument);
    EXPECT_EQ(tested.Size(), 1u);
    tested.Add(1, next);
    reference.Add(1, next);
    EXPECT_EQ(tested.GetStats().max_level, reference.GetStats().max_level);
    EXPECT_EQ(tested.GetStats().edges_level0, reference.GetStats().edges_level0);
    const float query[] = {0};
    float distances[2]{};
    ann::idx_t ids[2]{};
    tested.Search(query, 2, {distances, ids, 2});
    EXPECT_EQ(ids[0], 0u);
    EXPECT_EQ(ids[1], 1u);
}

TEST(HNSWIndex, SameSeedAndBatchingProduceSameResultsAndDegreeBounds) {
    constexpr size_t n = 80;
    constexpr size_t dim = 8;
    const auto base = UnitVectors(n, dim, 7);
    ann::HNSWParams params;
    params.M = 4;
    params.Mmax0 = 8;
    params.ef_construction = 32;
    ann::HNSWIndex batch(dim, ann::Metric::L2, params);
    ann::HNSWIndex incremental(dim, ann::Metric::L2, params);
    batch.Add(n, base.data());
    for (size_t i = 0; i < n; ++i) {
        incremental.Add(1, base.data() + i * dim);
    }
    const auto stats = batch.GetStats();
    EXPECT_EQ(stats.nodes, n);
    EXPECT_EQ(stats.max_level, incremental.GetStats().max_level);
    EXPECT_EQ(stats.edges_level0, incremental.GetStats().edges_level0);
    EXPECT_EQ(stats.edges_upper, incremental.GetStats().edges_upper);
    EXPECT_LE(stats.edges_level0, n * params.Mmax0);
    EXPECT_LE(stats.edges_upper, n * static_cast<size_t>(stats.max_level) * params.M);
    float first_dist[10]{};
    float second_dist[10]{};
    ann::idx_t first_ids[10]{};
    ann::idx_t second_ids[10]{};
    batch.Search(base.data(), 10, {first_dist, first_ids, 10});
    incremental.Search(base.data(), 10, {second_dist, second_ids, 10});
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(first_ids[i], second_ids[i]);
        EXPECT_FLOAT_EQ(first_dist[i], second_dist[i]);
    }
}

TEST(HNSWIndex, ConstQueriesAndSearchBatchPreserveIndexState) {
    const float base[] = {0, 1, 2, 3};
    ann::HNSWIndex index(1, ann::Metric::L2);
    index.Add(4, base);
    const auto bytes = index.MemoryUsage();
    const auto edges = index.GetStats().edges_level0;
    const ann::Index& read_only = index;
    const float queries[] = {0, 3};
    float distances[4]{};
    ann::idx_t ids[4]{};
    read_only.SearchBatch(queries, 2, 2, {distances, ids, 2});
    EXPECT_EQ(ids[0], 0u);
    EXPECT_EQ(ids[1], 1u);
    EXPECT_EQ(ids[2], 3u);
    EXPECT_EQ(ids[3], 2u);
    EXPECT_EQ(index.MemoryUsage(), bytes);
    EXPECT_EQ(index.GetStats().edges_level0, edges);
}

TEST(HNSWIndex, SmallRandomDataHasIndependentFlatRecallBaseline) {
    constexpr size_t n = 600;
    constexpr size_t nq = 40;
    constexpr size_t dim = 16;
    constexpr size_t k = 10;
    const auto base = UnitVectors(n, dim, 100);
    const auto queries = UnitVectors(nq, dim, 200);
    for (const auto metric : {ann::Metric::L2, ann::Metric::InnerProduct}) {
        ann::HNSWParams params;
        params.M = 12;
        params.Mmax0 = 24;
        params.ef_construction = 80;
        params.ef_search = 80;
        ann::HNSWIndex index(dim, metric, params);
        ann::FlatIndex flat(dim, metric);
        index.Add(n, base.data());
        flat.Add(n, base.data());
        size_t hits = 0;
        float got_dist[k]{};
        float ref_dist[k]{};
        ann::idx_t got[k]{};
        ann::idx_t ref[k]{};
        for (size_t q = 0; q < nq; ++q) {
            index.Search(queries.data() + q * dim, k, {got_dist, got, k});
            flat.Search(queries.data() + q * dim, k, {ref_dist, ref, k});
            for (const auto id : got) {
                if (id != ann::kInvalidIdx && std::find(ref, ref + k, id) != ref + k) {
                    ++hits;
                }
            }
        }
        const double recall = static_cast<double>(hits) / (nq * k);
        std::cout << "hnsw_small_recall metric=" << static_cast<unsigned>(metric) << " n=" << n
                  << " nq=" << nq << " dim=" << dim << " k=" << k << " recall=" << recall << '\n';
        EXPECT_GE(recall, 0.8);
        EXPECT_GE(index.MemoryUsage(), sizeof(index) + n * dim * sizeof(float));
    }
}

}  // namespace
