#include "hnsw_level.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

#include "hnsw_graph.h"

namespace {

using ann::detail::HNSWLevelFromUnit;
using ann::detail::HNSWLevelGenerator;
using ann::detail::HNSWOpenUnit;

TEST(HNSWLevel, RejectsNeighborCapacityBelowTwo) {
    EXPECT_THROW(HNSWLevelGenerator(0), std::invalid_argument);
    EXPECT_THROW(HNSWLevelGenerator(1), std::invalid_argument);
    EXPECT_NO_THROW(HNSWLevelGenerator(2));
}

TEST(HNSWLevel, IntegerMappingKeepsBothEndpointsOpen) {
    EXPECT_DOUBLE_EQ(HNSWOpenUnit(0), 0x1p-53);
    EXPECT_DOUBLE_EQ(HNSWOpenUnit(std::numeric_limits<uint64_t>::max()), 1.0 - 0x1p-53);
    EXPECT_GT(HNSWOpenUnit(0), 0.0);
    EXPECT_LT(HNSWOpenUnit(std::numeric_limits<uint64_t>::max()), 1.0);
    EXPECT_DOUBLE_EQ(HNSWOpenUnit(uint64_t{1} << 63), 0.5 + 0x1p-53);
}

TEST(HNSWLevel, FormulaMapsInteriorProbabilityIntervals) {
    for (const size_t m : {2u, 4u, 16u, 32u}) {
        for (uint8_t level = 0; level < 10; ++level) {
            // 选区间内部的半层位置，避免把数学边界附近的 libm 舍入当作错误。
            const double unit = std::pow(static_cast<double>(m), -(level + 0.5));
            EXPECT_EQ(HNSWLevelFromUnit(unit, 1.0 / std::log(static_cast<double>(m))), level);
        }
    }
    EXPECT_EQ(HNSWLevelFromUnit(std::nextafter(1.0, 0.0), 1.0), 0u);
}

TEST(HNSWLevel, PureConversionRejectsInvalidInputs) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const double unit : {0.0, 1.0, -0.5, 2.0, nan, inf}) {
        EXPECT_THROW(HNSWLevelFromUnit(unit, 1.0), std::invalid_argument);
    }
    for (const double multiplier : {0.0, -1.0, nan, inf}) {
        EXPECT_THROW(HNSWLevelFromUnit(0.5, multiplier), std::invalid_argument);
    }
}

TEST(HNSWLevel, TinyPositiveInputSaturatesWithoutIntegerWrap) {
    const double tiny = std::numeric_limits<double>::denorm_min();
    EXPECT_EQ(HNSWLevelFromUnit(tiny, 1.0 / std::log(2.0)), 255u);
    EXPECT_EQ(HNSWLevelFromUnit(0.25, std::numeric_limits<double>::max()), 255u);
}

TEST(HNSWLevel, SameSeedProducesSameSequence) {
    for (const size_t m : {2u, 16u, 32u}) {
        HNSWLevelGenerator first(m, 42);
        HNSWLevelGenerator second(m, 42);
        HNSWLevelGenerator default_seed(m);
        for (size_t i = 0; i < 4096; ++i) {
            const auto level = first.Next();
            EXPECT_EQ(second.Next(), level);
            EXPECT_EQ(default_seed.Next(), level);
        }
    }
}

TEST(HNSWLevel, DifferentSeedsProduceDifferentSequences) {
    HNSWLevelGenerator first(16, 42);
    HNSWLevelGenerator second(16, 43);
    bool differs = false;
    for (size_t i = 0; i < 4096; ++i) {
        differs = (first.Next() != second.Next()) || differs;
    }
    EXPECT_TRUE(differs);
}

TEST(HNSWLevel, InterleavedGeneratorsHaveIndependentState) {
    HNSWLevelGenerator reference(4, 17);
    HNSWLevelGenerator tested(4, 17);
    HNSWLevelGenerator unrelated(4, 99);
    for (size_t i = 0; i < 4096; ++i) {
        const auto unused = unrelated.Next();
        static_cast<void>(unused);
        EXPECT_EQ(tested.Next(), reference.Next());
    }
}

TEST(HNSWLevel, MillionSamplesMatchTheoreticalTailAndMean) {
    constexpr size_t samples = 1000000;
    for (const size_t m : {2u, 4u, 16u, 32u}) {
        SCOPED_TRACE(m);
        HNSWLevelGenerator generator(m, 42);
        std::array<size_t, 256> histogram{};
        double sum = 0.0;
        uint8_t max_seen = 0;
        for (size_t i = 0; i < samples; ++i) {
            const auto level = generator.Next();
            ++histogram[level];
            sum += level;
            max_seen = std::max(max_seen, level);
        }
        size_t tail = samples;
        double probability = 1.0;
        for (size_t level = 1; level <= max_seen; ++level) {
            tail -= histogram[level - 1];
            probability /= static_cast<double>(m);
            const double expected = samples * probability;
            if (expected >= 25.0) {
                // 固定种子避免随机抖动；按二项分布给 6σ 加离散余量。
                const double sigma = std::sqrt(samples * probability * (1.0 - probability));
                EXPECT_NEAR(static_cast<double>(tail), expected, 6.0 * sigma + 4.0);
            }
        }
        const double expected_mean = 1.0 / (static_cast<double>(m) - 1.0);
        const double mean_sigma =
            std::sqrt(static_cast<double>(m)) / (static_cast<double>(m) - 1.0) / std::sqrt(samples);
        EXPECT_NEAR(sum / samples, expected_mean, 6.0 * mean_sigma);
        // 保留实测分布到 LastTest.log，便于回看，不将其当成性能计时。
        std::cout << "level_distribution M=" << m << " samples=" << samples
                  << " mean=" << sum / samples << " expected_mean=" << expected_mean
                  << " ge1=" << samples - histogram[0]
                  << " ge2=" << samples - histogram[0] - histogram[1]
                  << " max_level=" << static_cast<unsigned>(max_seen) << '\n';
    }
}

TEST(HNSWLevel, GeneratedLayersIntegrateWithGraphEntryAndMetadata) {
    ann::detail::HNSWGraphStorage graph(2, 4, 8);
    HNSWLevelGenerator generator(4, 42);
    int expected_max = -1;
    ann::idx_t expected_entry = ann::kInvalidIdx;
    for (ann::idx_t i = 0; i < 5000; ++i) {
        const auto level = generator.Next();
        const std::array<float, 2> vector{static_cast<float>(i), 0};
        ASSERT_EQ(graph.Append(vector, level), i);
        EXPECT_EQ(graph.Level(i), level);
        if (level > expected_max) {
            expected_max = level;
            expected_entry = i;
        }
        EXPECT_EQ(graph.MaxLevel(), expected_max);
        EXPECT_EQ(graph.EntryPoint(), expected_entry);
        EXPECT_TRUE(graph.Neighbors(i, level).empty());
    }
    EXPECT_GT(expected_max, 0);
    EXPECT_EQ(graph.Size(), 5000u);
}

}  // namespace
