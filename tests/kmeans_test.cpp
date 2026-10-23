#include "ann/kmeans.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "kmeans_internal.h"

namespace {

TEST(KMeans, EmptyClustersChooseFarthestPointsSequentially) {
    const float data[] = {0, 4, 8, 10};
    float centers[] = {0, 999, 999};
    const size_t counts[] = {4, 0, 0};
    double nearest[4]{};
    unsigned char chosen[4]{};
    ann::detail::ReseedEmptyKMeansClusters(4, 1, data, centers, counts, nearest, chosen);
    EXPECT_FLOAT_EQ(centers[0], 0);
    EXPECT_FLOAT_EQ(centers[1], 10);
    EXPECT_FLOAT_EQ(centers[2], 4);
    // 若第二次只跳过已取样本却不更新最近距离，会错误地选择 8。
    EXPECT_DOUBLE_EQ(nearest[1], 0);
    EXPECT_DOUBLE_EQ(nearest[2], 4);
    EXPECT_DOUBLE_EQ(nearest[3], 0);
}

TEST(KMeans, SingleClusterHandComputedMean) {
    ann::KMeans trainer(2, 1);
    const float data[] = {0, 2, 2, 4, 4, 6};
    float centers[2]{};
    trainer.Train(3, data, centers);
    EXPECT_FLOAT_EQ(centers[0], 2);
    EXPECT_FLOAT_EQ(centers[1], 4);
    EXPECT_EQ(trainer.ClusterSizes(), (std::vector<size_t>{3}));
}

TEST(KMeans, SeparatedClustersHaveExpectedMeans) {
    ann::KMeans trainer(2, 2);
    const float data[] = {0, 0, 0, 2, 10, 10, 10, 12};
    float centers[4]{};
    trainer.Train(4, data, centers);
    const size_t low = centers[0] < centers[2] ? 0 : 2;
    const size_t high = 2 - low;
    EXPECT_FLOAT_EQ(centers[low], 0);
    EXPECT_FLOAT_EQ(centers[low + 1], 1);
    EXPECT_FLOAT_EQ(centers[high], 10);
    EXPECT_FLOAT_EQ(centers[high + 1], 11);
    EXPECT_EQ(trainer.ClusterSizes(), (std::vector<size_t>{2, 2}));
}

TEST(KMeans, AssignUsesL2AndLowerClusterIdForTies) {
    const ann::KMeans trainer(2, 3);
    const float centers[] = {0, 0, 4, 0, 0, 4};
    const float data[] = {2, 0, 0, 3, 4, 1};
    uint32_t assignments[3]{};
    trainer.Assign(3, data, centers, assignments);
    EXPECT_EQ(assignments[0], 0U);
    EXPECT_EQ(assignments[1], 2U);
    EXPECT_EQ(assignments[2], 1U);
    EXPECT_TRUE(trainer.ClusterSizes().empty());
}

TEST(KMeans, OneClusterPerDistinctPoint) {
    ann::KMeans trainer(1, 4, 2);
    const float data[] = {-9, -1, 4, 20};
    float centers[4]{};
    trainer.Train(4, data, centers);
    std::sort(std::begin(centers), std::end(centers));
    EXPECT_TRUE(std::equal(std::begin(data), std::end(data), std::begin(centers)));
    EXPECT_EQ(trainer.ClusterSizes(), (std::vector<size_t>{1, 1, 1, 1}));
}

TEST(KMeans, DuplicatePointsAllowFinalEmptyClusters) {
    ann::KMeans trainer(2, 4, 3);
    const float data[] = {2, -3, 2, -3, 2, -3, 2, -3, 2, -3};
    float centers[8]{};
    trainer.Train(5, data, centers);
    for (size_t c = 0; c < 4; ++c) {
        EXPECT_FLOAT_EQ(centers[c * 2], 2);
        EXPECT_FLOAT_EQ(centers[c * 2 + 1], -3);
    }
    EXPECT_EQ(trainer.ClusterSizes(), (std::vector<size_t>{5, 0, 0, 0}));
}

TEST(KMeans, FinalCountsMatchDeliveredCentroidsAfterOneIteration) {
    const float data[] = {-20, -10, -9, -8, 0, 1, 2, 3, 30, 40, 41};
    ann::KMeans trainer(1, 4, 1, 17);
    float centers[4]{};
    uint32_t assignments[11]{};
    trainer.Train(11, data, centers);
    trainer.Assign(11, data, centers, assignments);
    std::vector<size_t> counts(4);
    for (uint32_t c : assignments) {
        ASSERT_LT(c, 4U);
        ++counts[c];
    }
    EXPECT_EQ(counts, trainer.ClusterSizes());
    EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), size_t{0}), 11U);
}

TEST(KMeans, SameSeedReproducesTrainingSequence) {
    const float data[] = {-10, -8, -3, 0, 1, 3, 8, 12};
    ann::KMeans first(1, 3, 2, 123), second(1, 3, 2, 123);
    float a[3]{}, b[3]{};
    for (int repetition = 0; repetition < 3; ++repetition) {
        first.Train(8, data, a);
        second.Train(8, data, b);
        EXPECT_TRUE(std::equal(std::begin(a), std::end(a), std::begin(b)));
        EXPECT_EQ(first.ClusterSizes(), second.ClusterSizes());
    }
}

TEST(KMeans, FiniteExtremeInputsDoNotOverflowFloatArithmetic) {
    const float limit = std::numeric_limits<float>::max();
    const float data[] = {limit, limit, -limit, -limit};
    ann::KMeans trainer(1, 1);
    float center{};
    trainer.Train(4, data, &center);
    EXPECT_FLOAT_EQ(center, 0);
    const float centers[] = {-limit, limit};
    uint32_t assignments[4]{};
    ann::KMeans(1, 2).Assign(4, data, centers, assignments);
    EXPECT_EQ(assignments[0], 1U);
    EXPECT_EQ(assignments[2], 0U);
}

TEST(KMeans, TrainingOutputCanOverlapInput) {
    ann::KMeans trainer(1, 1);
    float data[] = {2, 4, 6};
    trainer.Train(3, data, data);
    EXPECT_FLOAT_EQ(data[0], 4);
}

TEST(KMeans, RejectsInvalidShapesAndParameters) {
    EXPECT_THROW(ann::KMeans(0, 1), std::invalid_argument);
    EXPECT_THROW(ann::KMeans(1, 0), std::invalid_argument);
    EXPECT_THROW(ann::KMeans(1, 1, 0), std::invalid_argument);
    EXPECT_THROW(ann::KMeans(1, static_cast<size_t>(UINT32_MAX) + 1), std::invalid_argument);
    EXPECT_THROW(ann::KMeans(std::numeric_limits<size_t>::max(), 2), std::length_error);
    ann::KMeans trainer(1, 2);
    const float data[] = {1, 2};
    float centers[2]{};
    uint32_t assignment{};
    EXPECT_THROW(trainer.Train(1, data, centers), std::invalid_argument);
    EXPECT_THROW(trainer.Train(2, nullptr, centers), std::invalid_argument);
    EXPECT_THROW(trainer.Train(2, data, nullptr), std::invalid_argument);
    EXPECT_THROW(trainer.Assign(1, data, centers, nullptr), std::invalid_argument);
    EXPECT_THROW(trainer.Assign(1, nullptr, centers, &assignment), std::invalid_argument);
    EXPECT_THROW(trainer.Assign(1, data, nullptr, &assignment), std::invalid_argument);
    EXPECT_THROW(trainer.Assign(std::numeric_limits<size_t>::max(), data, centers, &assignment),
                 std::length_error);
    EXPECT_NO_THROW(trainer.Assign(0, nullptr, nullptr, nullptr));
}

TEST(KMeans, InvalidTrainingPreservesOutputStatisticsAndRandomState) {
    ann::KMeans trainer(1, 2, 3), reference(1, 2, 3);
    const float data[] = {0, 1, 10, 11};
    float centers[2]{}, expected[2]{};
    trainer.Train(4, data, centers);
    reference.Train(4, data, expected);
    const auto old_counts = trainer.ClusterSizes();
    const float bad[] = {0, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(trainer.Train(2, bad, centers), std::invalid_argument);
    EXPECT_TRUE(std::equal(std::begin(centers), std::end(centers), std::begin(expected)));
    EXPECT_EQ(trainer.ClusterSizes(), old_counts);
    trainer.Train(4, data, centers);
    reference.Train(4, data, expected);
    EXPECT_TRUE(std::equal(std::begin(centers), std::end(centers), std::begin(expected)));
    uint32_t assignments[2] = {99, 99};
    const float infinite[] = {0, std::numeric_limits<float>::infinity()};
    EXPECT_THROW(trainer.Assign(2, infinite, centers, assignments), std::invalid_argument);
    EXPECT_THROW(trainer.Assign(2, data, infinite, assignments), std::invalid_argument);
    EXPECT_EQ(assignments[0], 99U);
}

}  // namespace
