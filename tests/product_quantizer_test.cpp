#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "ann/quantizer.h"

namespace {

// 两个二维子空间各有两个精确中心，四种组合全部出现。
const float kGrid[] = {0, 0, 10, 10, 0, 0, 12, 12, 2, 2, 10, 10, 2, 2, 12, 12};

TEST(ProductQuantizer, HandComputedSubspacesAndRoundTrip) {
    ann::ProductQuantizer quantizer(4, 2, 2);
    quantizer.Train(4, kGrid);
    for (size_t i = 0; i < 4; ++i) {
        uint8_t code[2]{};
        float decoded[4]{};
        quantizer.Encode(kGrid + i * 4, code);
        quantizer.Decode(code, decoded);
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_FLOAT_EQ(decoded[d], kGrid[i * 4 + d]);
        }
    }
    EXPECT_EQ(quantizer.CodeSize(), 2U);
    EXPECT_EQ(quantizer.LutSize(), 4U);
    EXPECT_STREQ(quantizer.Name(), "PQ");
}

TEST(ProductQuantizer, LUTLayoutAndUnquantizedQueryAreHandComputed) {
    ann::ProductQuantizer quantizer(4, 2, 2);
    quantizer.Train(4, kGrid);
    const float query[] = {3, 3, 13, 13};
    float lut[4]{};
    quantizer.BuildLUT(query, lut);
    EXPECT_FLOAT_EQ(std::min(lut[0], lut[1]), 2);
    EXPECT_FLOAT_EQ(std::max(lut[0], lut[1]), 18);
    EXPECT_FLOAT_EQ(std::min(lut[2], lut[3]), 2);
    EXPECT_FLOAT_EQ(std::max(lut[2], lut[3]), 18);
    uint8_t first[2]{}, last[2]{};
    quantizer.Encode(kGrid, first);
    quantizer.Encode(kGrid + 12, last);
    EXPECT_FLOAT_EQ(quantizer.DistanceFromLUT(lut, first), 36);
    EXPECT_FLOAT_EQ(quantizer.DistanceFromLUT(lut, last), 4);
}

TEST(ProductQuantizer, EquidistantEncodingUsesLowerCodeId) {
    ann::ProductQuantizer quantizer(4, 2, 2);
    quantizer.Train(4, kGrid);
    const float middle[] = {1, 1, 11, 11};
    uint8_t code[] = {99, 99};
    quantizer.Encode(middle, code);
    EXPECT_EQ(code[0], 0);
    EXPECT_EQ(code[1], 0);
}

TEST(ProductQuantizer, AllCartesianCodeCombinationsDecodeCorrectly) {
    ann::ProductQuantizer quantizer(4, 2, 2);
    quantizer.Train(4, kGrid);
    for (uint8_t first = 0; first < 2; ++first) {
        for (uint8_t second = 0; second < 2; ++second) {
            const uint8_t code[] = {first, second};
            float decoded[4]{};
            quantizer.Decode(code, decoded);
            EXPECT_FLOAT_EQ(decoded[0], decoded[1]);
            EXPECT_FLOAT_EQ(decoded[2], decoded[3]);
            EXPECT_FLOAT_EQ(decoded[0], decoded[0] < 1 ? 0.0f : 2.0f);
            EXPECT_FLOAT_EQ(decoded[2], decoded[2] < 11 ? 10.0f : 12.0f);
            uint8_t encoded[2]{};
            quantizer.Encode(decoded, encoded);
            EXPECT_EQ(encoded[0], first);
            EXPECT_EQ(encoded[1], second);
        }
    }
}

TEST(ProductQuantizer, RandomADCMatchesIndependentDecodedDoubleReference) {
    constexpr size_t dim = 16, segments = 4, count = 40;
    ann::ProductQuantizer quantizer(dim, segments, 8, 17);
    std::mt19937 rng(923);
    std::uniform_real_distribution<float> coordinate(-10, 10);
    std::vector<float> training(count * dim);
    for (auto& value : training) {
        value = coordinate(rng);
    }
    quantizer.Train(count, training.data());
    std::vector<float> lut(quantizer.LutSize());
    for (size_t repeat = 0; repeat < 50; ++repeat) {
        float point[dim]{}, query[dim]{}, decoded[dim]{};
        uint8_t code[segments]{};
        for (size_t d = 0; d < dim; ++d) {
            point[d] = coordinate(rng);
            query[d] = coordinate(rng);
        }
        quantizer.Encode(point, code);
        quantizer.Decode(code, decoded);
        quantizer.BuildLUT(query, lut.data());
        double reference = 0;
        for (size_t d = 0; d < dim; ++d) {
            const double delta = static_cast<double>(query[d]) - static_cast<double>(decoded[d]);
            reference += delta * delta;
        }
        EXPECT_NEAR(static_cast<double>(quantizer.DistanceFromLUT(lut.data(), code)),
                    reference,
                    std::max(1.0, reference) * 2e-7);
    }
}

TEST(ProductQuantizer, SingleSegmentSingleCenterAndSingleCoordinateSegments) {
    ann::ProductQuantizer single(3, 1, 1);
    const float training[] = {-2, 0, 4, 3, 5, 9};
    single.Train(2, training);
    const uint8_t code = 0;
    float decoded[3]{};
    single.Decode(&code, decoded);
    EXPECT_FLOAT_EQ(decoded[0], 0.5f);
    EXPECT_FLOAT_EQ(decoded[1], 2.5f);
    EXPECT_FLOAT_EQ(decoded[2], 6.5f);
    const float query[] = {0, 0, 0};
    float lut{};
    single.BuildLUT(query, &lut);
    EXPECT_FLOAT_EQ(lut, 48.75f);

    ann::ProductQuantizer scalar_segments(2, 2, 3);
    const float rows[] = {0, -4, 1, 5, 1000, 6};
    scalar_segments.Train(3, rows);
    for (size_t i = 0; i < 3; ++i) {
        uint8_t encoded[2]{};
        float restored[2]{};
        scalar_segments.Encode(rows + i * 2, encoded);
        scalar_segments.Decode(encoded, restored);
        EXPECT_FLOAT_EQ(restored[0], rows[i * 2]);
        EXPECT_FLOAT_EQ(restored[1], rows[i * 2 + 1]);
    }
}

TEST(ProductQuantizer, Default256CentersIncludeHighestByteCode) {
    ann::ProductQuantizer quantizer(2, 2);
    std::vector<float> training(256 * 2);
    for (size_t i = 0; i < 256; ++i) {
        training[i * 2] = static_cast<float>(i);
        training[i * 2 + 1] = static_cast<float>(1000 + i);
    }
    quantizer.Train(256, training.data());
    bool seen[2][256]{};
    for (size_t i = 0; i < 256; ++i) {
        uint8_t code[2]{};
        float decoded[2]{};
        quantizer.Encode(training.data() + i * 2, code);
        quantizer.Decode(code, decoded);
        for (size_t s = 0; s < 2; ++s) {
            seen[s][code[s]] = true;
            EXPECT_FLOAT_EQ(decoded[s], training[i * 2 + s]);
        }
    }
    for (size_t c = 0; c < 256; ++c) {
        EXPECT_TRUE(seen[0][c]);
        EXPECT_TRUE(seen[1][c]);
    }
    const uint8_t maximum[] = {255, 255};
    float decoded[2]{};
    EXPECT_NO_THROW(quantizer.Decode(maximum, decoded));
    EXPECT_EQ(quantizer.LutSize(), 512U);
}

TEST(ProductQuantizer, DuplicateTrainingVectorsAllowDuplicateCodewords) {
    ann::ProductQuantizer quantizer(4, 2, 4);
    const float rows[] = {2, -3, 5, 8, 2, -3, 5, 8, 2, -3, 5, 8, 2, -3, 5, 8};
    quantizer.Train(4, rows);
    uint8_t code[2]{};
    quantizer.Encode(rows, code);
    EXPECT_EQ(code[0], 0);
    EXPECT_EQ(code[1], 0);
    const uint8_t other[] = {3, 3};
    float decoded[4]{};
    quantizer.Decode(other, decoded);
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_FLOAT_EQ(decoded[d], rows[d]);
    }
}

TEST(ProductQuantizer, FixedSeedAndRepeatedTrainingReproduceModels) {
    ann::ProductQuantizer first(4, 2, 2, 123), second(4, 2, 2, 123);
    first.Train(4, kGrid);
    second.Train(4, kGrid);
    const float query[] = {1, -2, 13, 7};
    float a[4]{}, b[4]{}, repeated[4]{};
    first.BuildLUT(query, a);
    second.BuildLUT(query, b);
    first.Train(4, kGrid);
    first.BuildLUT(query, repeated);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]);
        EXPECT_FLOAT_EQ(a[i], repeated[i]);
    }
}

TEST(ProductQuantizer, RetrainingReplacesAllSubspaceCodebooks) {
    ann::ProductQuantizer quantizer(4, 2, 2);
    quantizer.Train(4, kGrid);
    float shifted[16]{};
    for (size_t i = 0; i < 16; ++i) {
        shifted[i] = kGrid[i] + 100;
    }
    quantizer.Train(4, shifted);
    uint8_t code[2]{};
    float decoded[4]{};
    quantizer.Encode(shifted, code);
    quantizer.Decode(code, decoded);
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_FLOAT_EQ(decoded[d], shifted[d]);
    }
    const float bad[] = {0, 0, 0, 0, 0, 0, 0, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(quantizer.Train(2, bad), std::invalid_argument);
    float after[4]{};
    quantizer.Decode(code, after);
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_FLOAT_EQ(after[d], decoded[d]);
    }
}

TEST(ProductQuantizer, CallerLUTsRemainIndependentAcrossQueries) {
    ann::ProductQuantizer quantizer(4, 2, 2);
    quantizer.Train(4, kGrid);
    const auto& model = quantizer;
    const size_t memory = model.MemoryUsage();
    const float a[] = {0, 0, 10, 10}, b[] = {5, 5, 15, 15};
    float first[4]{}, second[4]{};
    model.BuildLUT(a, first);
    const std::vector<float> saved(std::begin(first), std::end(first));
    model.BuildLUT(b, second);
    uint8_t code[2]{};
    model.Encode(a, code);
    EXPECT_FLOAT_EQ(model.DistanceFromLUT(first, code), 0);
    EXPECT_FLOAT_EQ(model.DistanceFromLUT(second, code), 100);
    EXPECT_TRUE(std::equal(saved.begin(), saved.end(), std::begin(first)));
    EXPECT_EQ(model.MemoryUsage(), memory);
}

TEST(ProductQuantizer, RejectsMalformedShapeAndTrainingInput) {
    EXPECT_THROW(ann::ProductQuantizer(0, 1), std::invalid_argument);
    EXPECT_THROW(ann::ProductQuantizer(4, 0), std::invalid_argument);
    EXPECT_THROW(ann::ProductQuantizer(5, 2), std::invalid_argument);
    EXPECT_THROW(ann::ProductQuantizer(4, 8), std::invalid_argument);
    EXPECT_THROW(ann::ProductQuantizer(4, 2, 0), std::invalid_argument);
    EXPECT_THROW(ann::ProductQuantizer(4, 2, 257), std::invalid_argument);
    EXPECT_THROW(ann::ProductQuantizer(std::numeric_limits<size_t>::max(), 1, 2),
                 std::length_error);
    ann::ProductQuantizer quantizer(4, 2, 2);
    EXPECT_THROW(quantizer.Train(1, kGrid), std::invalid_argument);
    EXPECT_THROW(quantizer.Train(2, nullptr), std::invalid_argument);
    EXPECT_THROW(quantizer.Train(std::numeric_limits<size_t>::max(), kGrid), std::length_error);
    float bad[8]{};
    bad[7] = std::numeric_limits<float>::infinity();
    EXPECT_THROW(quantizer.Train(2, bad), std::invalid_argument);
}

TEST(ProductQuantizer, RequiresTrainingAndValidBuffersAndCodes) {
    ann::ProductQuantizer quantizer(4, 2, 2);
    uint8_t code[] = {0, 0};
    float decoded[] = {-7, -7, -7, -7};
    float lut[] = {-7, -7, -7, -7};
    EXPECT_THROW(quantizer.Encode(kGrid, code), std::logic_error);
    EXPECT_THROW(quantizer.Decode(code, decoded), std::logic_error);
    EXPECT_THROW(quantizer.BuildLUT(kGrid, lut), std::logic_error);
    EXPECT_THROW(quantizer.DistanceFromLUT(lut, code), std::logic_error);
    quantizer.Train(4, kGrid);
    EXPECT_THROW(quantizer.Encode(nullptr, code), std::invalid_argument);
    EXPECT_THROW(quantizer.Encode(kGrid, nullptr), std::invalid_argument);
    EXPECT_THROW(quantizer.Decode(nullptr, decoded), std::invalid_argument);
    EXPECT_THROW(quantizer.Decode(code, nullptr), std::invalid_argument);
    EXPECT_THROW(quantizer.BuildLUT(nullptr, lut), std::invalid_argument);
    EXPECT_THROW(quantizer.BuildLUT(kGrid, nullptr), std::invalid_argument);
    EXPECT_THROW(quantizer.DistanceFromLUT(nullptr, code), std::invalid_argument);
    EXPECT_THROW(quantizer.DistanceFromLUT(lut, nullptr), std::invalid_argument);
    const uint8_t invalid[] = {0, 2};
    EXPECT_THROW(quantizer.Decode(invalid, decoded), std::invalid_argument);
    EXPECT_FLOAT_EQ(decoded[0], -7);
    EXPECT_THROW(quantizer.DistanceFromLUT(lut, invalid), std::invalid_argument);
    const float bad[] = {0, 0, 0, std::numeric_limits<float>::quiet_NaN()};
    code[0] = 99;
    EXPECT_THROW(quantizer.Encode(bad, code), std::invalid_argument);
    EXPECT_EQ(code[0], 99);
    EXPECT_THROW(quantizer.BuildLUT(bad, lut), std::invalid_argument);
    EXPECT_FLOAT_EQ(lut[0], -7);
}

TEST(ProductQuantizer, FloatLUTAndTotalOverflowDoNotBecomeInvalidDistances) {
    ann::ProductQuantizer quantizer(2, 2, 1);
    const float row[] = {0, 0};
    quantizer.Train(1, row);
    const float query[] = {1, std::numeric_limits<float>::max()};
    float lut[] = {-7, -7};
    EXPECT_THROW(quantizer.BuildLUT(query, lut), std::overflow_error);
    EXPECT_FLOAT_EQ(lut[0], -7);
    EXPECT_FLOAT_EQ(lut[1], -7);
    const uint8_t code[] = {0, 0};
    lut[0] = std::numeric_limits<float>::max();
    lut[1] = lut[0];
    EXPECT_THROW(quantizer.DistanceFromLUT(lut, code), std::overflow_error);
    lut[0] = -1;
    EXPECT_THROW(quantizer.DistanceFromLUT(lut, code), std::invalid_argument);
    lut[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(quantizer.DistanceFromLUT(lut, code), std::invalid_argument);
}

TEST(ProductQuantizer, WideFiniteVectorsUseDoubleEncodingDistances) {
    ann::ProductQuantizer quantizer(1, 1, 2);
    const float limit = std::numeric_limits<float>::max();
    const float rows[] = {-limit, limit};
    quantizer.Train(2, rows);
    uint8_t code{};
    float decoded{};
    quantizer.Encode(&limit, &code);
    quantizer.Decode(&code, &decoded);
    EXPECT_FLOAT_EQ(decoded, limit);
}

TEST(ProductQuantizer, ModelCapacityExcludesCallerCodesAndLUT) {
    ann::ProductQuantizer quantizer(4, 2, 2);
    const size_t before = quantizer.MemoryUsage();
    EXPECT_GE(before, sizeof(quantizer));
    quantizer.Train(4, kGrid);
    EXPECT_GE(quantizer.MemoryUsage(), sizeof(quantizer) + 2 * 4 * sizeof(float));
    EXPECT_EQ(quantizer.CodeSize(), 2U);
    EXPECT_EQ(quantizer.LutSize(), 4U);
}

}  // namespace
