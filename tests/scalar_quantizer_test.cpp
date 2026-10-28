#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "ann/quantizer.h"

namespace {

TEST(ScalarQuantizer, HandComputedTrainingAndEncoding) {
    ann::ScalarQuantizer quantizer(2);
    const float training[] = {-1, 10, 1, 20};
    quantizer.Train(2, training);
    const float point[] = {0, 15};
    uint8_t code[2]{};
    float decoded[2]{};
    quantizer.Encode(point, code);
    EXPECT_EQ(code[0], 128);
    EXPECT_EQ(code[1], 128);
    quantizer.Decode(code, decoded);
    EXPECT_FLOAT_EQ(decoded[0], static_cast<float>(1.0 / 255.0));
    EXPECT_FLOAT_EQ(decoded[1], static_cast<float>(10.0 + 1280.0 / 255.0));
    EXPECT_EQ(quantizer.CodeSize(), 2U);
    EXPECT_EQ(quantizer.LutSize(), 512U);
    EXPECT_STREQ(quantizer.Name(), "SQ8");
}

TEST(ScalarQuantizer, PreservesTrainingEndpoints) {
    ann::ScalarQuantizer quantizer(3);
    const float training[] = {-5, 2, -30, 5, 8, -10};
    quantizer.Train(2, training);
    for (size_t row = 0; row < 2; ++row) {
        uint8_t code[3]{};
        float decoded[3]{};
        quantizer.Encode(training + row * 3, code);
        quantizer.Decode(code, decoded);
        for (size_t d = 0; d < 3; ++d) {
            EXPECT_EQ(code[d], row == 0 ? 0 : 255);
            EXPECT_FLOAT_EQ(decoded[d], training[row * 3 + d]);
        }
    }
}

TEST(ScalarQuantizer, RoundsHalfStepsUpAndClipsOutOfRange) {
    ann::ScalarQuantizer quantizer(1);
    const float training[] = {0, 255};
    quantizer.Train(2, training);
    const float values[] = {-100, 0.49f, 0.5f, 1.5f, 254.5f, 1000};
    const uint8_t expected[] = {0, 0, 1, 2, 255, 255};
    for (size_t i = 0; i < 6; ++i) {
        uint8_t code{};
        quantizer.Encode(values + i, &code);
        EXPECT_EQ(code, expected[i]);
    }
}

TEST(ScalarQuantizer, ConstantDimensionsDecodeEveryCodeToConstant) {
    ann::ScalarQuantizer quantizer(2);
    const float training[] = {5, -2, 5, -2};
    quantizer.Train(2, training);
    const float point[] = {100, -100};
    uint8_t code[2]{};
    quantizer.Encode(point, code);
    EXPECT_EQ(code[0], 0);
    EXPECT_EQ(code[1], 0);
    code[0] = 255;
    code[1] = 72;
    float decoded[2]{};
    quantizer.Decode(code, decoded);
    EXPECT_FLOAT_EQ(decoded[0], 5);
    EXPECT_FLOAT_EQ(decoded[1], -2);
    const float query[] = {8, 2};
    std::vector<float> lut(quantizer.LutSize());
    quantizer.BuildLUT(query, lut.data());
    EXPECT_FLOAT_EQ(quantizer.DistanceFromLUT(lut.data(), code), 25);
}

TEST(ScalarQuantizer, HandComputedLUTLayoutAndAsymmetricQuery) {
    ann::ScalarQuantizer quantizer(2);
    const float training[] = {0, 0, 255, 510};
    quantizer.Train(2, training);
    // 查询在训练区间之外，必须保留原值，不对查询截断或编码。
    const float query[] = {-2, 600};
    std::vector<float> lut(quantizer.LutSize());
    quantizer.BuildLUT(query, lut.data());
    EXPECT_FLOAT_EQ(lut[0], 4);
    EXPECT_FLOAT_EQ(lut[255], 66049);
    EXPECT_FLOAT_EQ(lut[256], 360000);
    EXPECT_FLOAT_EQ(lut[511], 8100);
    const uint8_t code[] = {3, 4};
    EXPECT_FLOAT_EQ(lut[3], 25);
    EXPECT_FLOAT_EQ(lut[260], 350464);
    EXPECT_FLOAT_EQ(quantizer.DistanceFromLUT(lut.data(), code), 350489);
}

TEST(ScalarQuantizer, RandomADCMatchesIndependentDecodedL2) {
    constexpr size_t dim = 7;
    ann::ScalarQuantizer quantizer(dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> input(-10, 10);
    std::vector<float> training(40 * dim);
    for (auto& value : training) {
        value = input(rng);
    }
    quantizer.Train(40, training.data());
    std::vector<float> lut(quantizer.LutSize());
    for (size_t iteration = 0; iteration < 50; ++iteration) {
        float point[dim], query[dim], decoded[dim];
        uint8_t code[dim];
        for (size_t d = 0; d < dim; ++d) {
            point[d] = input(rng);
            query[d] = input(rng);
        }
        quantizer.Encode(point, code);
        quantizer.Decode(code, decoded);
        quantizer.BuildLUT(query, lut.data());
        double expected = 0;
        for (size_t d = 0; d < dim; ++d) {
            const double diff = static_cast<double>(query[d]) - static_cast<double>(decoded[d]);
            expected += diff * diff;
        }
        const double actual = quantizer.DistanceFromLUT(lut.data(), code);
        EXPECT_NEAR(actual, expected, std::max(1.0, expected) * 2e-7);
    }
}

TEST(ScalarQuantizer, ReconstructionErrorBoundInsideTrainingRange) {
    ann::ScalarQuantizer quantizer(1);
    const float training[] = {-3, 6};
    quantizer.Train(2, training);
    const double half_step = 9.0 / 510.0;
    for (size_t i = 0; i <= 1000; ++i) {
        const float value = static_cast<float>(-3.0 + 9.0 * static_cast<double>(i) / 1000.0);
        uint8_t code{};
        float decoded{};
        quantizer.Encode(&value, &code);
        quantizer.Decode(&code, &decoded);
        EXPECT_LE(std::abs(static_cast<double>(value) - static_cast<double>(decoded)),
                  half_step + 1e-6);
    }
}

TEST(ScalarQuantizer, WideFiniteRangeDecodesMonotonicallyWithoutOverflow) {
    ann::ScalarQuantizer quantizer(1);
    const float limit = std::numeric_limits<float>::max();
    const float training[] = {-limit, limit};
    quantizer.Train(2, training);
    float previous = -limit;
    for (size_t c = 0; c < 256; ++c) {
        const auto code = static_cast<uint8_t>(c);
        float decoded{};
        quantizer.Decode(&code, &decoded);
        EXPECT_TRUE(std::isfinite(decoded));
        EXPECT_GE(decoded, previous);
        EXPECT_LE(decoded, limit);
        previous = decoded;
    }
    uint8_t code{};
    quantizer.Encode(&limit, &code);
    EXPECT_EQ(code, 255);
    float decoded{};
    quantizer.Decode(&code, &decoded);
    EXPECT_FLOAT_EQ(decoded, limit);
}

TEST(ScalarQuantizer, SubnormalRangeDoesNotBecomeConstantByScaleUnderflow) {
    ann::ScalarQuantizer quantizer(1);
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float training[] = {0, tiny};
    quantizer.Train(2, training);
    uint8_t code{};
    quantizer.Encode(&tiny, &code);
    EXPECT_EQ(code, 255);
    float decoded{};
    quantizer.Decode(&code, &decoded);
    EXPECT_EQ(decoded, tiny);
}

TEST(ScalarQuantizer, RetrainingReplacesModelAndFailurePreservesIt) {
    ann::ScalarQuantizer quantizer(1);
    const float first[] = {0, 255};
    const float second[] = {-255, 0};
    const float bad[] = {0, std::numeric_limits<float>::quiet_NaN()};
    quantizer.Train(2, first);
    quantizer.Train(2, second);
    const size_t memory = quantizer.MemoryUsage();
    EXPECT_THROW(quantizer.Train(2, bad), std::invalid_argument);
    const uint8_t code = 255;
    float decoded = 123;
    quantizer.Decode(&code, &decoded);
    EXPECT_FLOAT_EQ(decoded, 0);
    EXPECT_EQ(quantizer.MemoryUsage(), memory);
}

TEST(ScalarQuantizer, SeparateCallerLUTsDoNotShareQueryState) {
    ann::ScalarQuantizer quantizer(1);
    const float training[] = {0, 255};
    quantizer.Train(2, training);
    const ann::ScalarQuantizer& read_only = quantizer;
    const float first = 0, second = 10;
    std::vector<float> a(read_only.LutSize()), b(read_only.LutSize());
    read_only.BuildLUT(&first, a.data());
    const auto saved = a;
    read_only.BuildLUT(&second, b.data());
    const uint8_t code = 3;
    EXPECT_FLOAT_EQ(read_only.DistanceFromLUT(a.data(), &code), 9);
    EXPECT_FLOAT_EQ(read_only.DistanceFromLUT(b.data(), &code), 49);
    EXPECT_EQ(a, saved);
}

TEST(ScalarQuantizer, RequiresTrainingAndRejectsEmptyOrInvalidTraining) {
    ann::ScalarQuantizer quantizer(1);
    const float point = 0;
    uint8_t code{};
    float output{};
    std::vector<float> lut(quantizer.LutSize());
    EXPECT_THROW(quantizer.Encode(&point, &code), std::logic_error);
    EXPECT_THROW(quantizer.Decode(&code, &output), std::logic_error);
    EXPECT_THROW(quantizer.BuildLUT(&point, lut.data()), std::logic_error);
    EXPECT_THROW(quantizer.DistanceFromLUT(lut.data(), &code), std::logic_error);
    EXPECT_THROW(quantizer.Train(0, &point), std::invalid_argument);
    EXPECT_THROW(quantizer.Train(1, nullptr), std::invalid_argument);
    EXPECT_THROW(quantizer.Train(std::numeric_limits<size_t>::max(), &point), std::length_error);
    const float infinity = std::numeric_limits<float>::infinity();
    EXPECT_THROW(quantizer.Train(1, &infinity), std::invalid_argument);
    EXPECT_THROW(quantizer.Decode(&code, &output), std::logic_error);
}

TEST(ScalarQuantizer, InvalidBuffersAndInputsDoNotPartiallyWriteOutputs) {
    ann::ScalarQuantizer quantizer(2);
    const float training[] = {0, 0, 1, 1};
    quantizer.Train(2, training);
    const float bad[] = {0, std::numeric_limits<float>::quiet_NaN()};
    uint8_t code[] = {99, 99};
    float decoded[2]{};
    std::vector<float> lut(quantizer.LutSize(), -7);
    EXPECT_THROW(quantizer.Encode(bad, code), std::invalid_argument);
    EXPECT_EQ(code[0], 99);
    EXPECT_THROW(quantizer.Encode(nullptr, code), std::invalid_argument);
    EXPECT_THROW(quantizer.Encode(training, nullptr), std::invalid_argument);
    EXPECT_THROW(quantizer.Decode(nullptr, decoded), std::invalid_argument);
    EXPECT_THROW(quantizer.Decode(code, nullptr), std::invalid_argument);
    EXPECT_THROW(quantizer.BuildLUT(bad, lut.data()), std::invalid_argument);
    EXPECT_EQ(lut.front(), -7);
    EXPECT_EQ(lut.back(), -7);
    EXPECT_THROW(quantizer.BuildLUT(nullptr, lut.data()), std::invalid_argument);
    EXPECT_THROW(quantizer.BuildLUT(training, nullptr), std::invalid_argument);
    EXPECT_THROW(quantizer.DistanceFromLUT(nullptr, code), std::invalid_argument);
    EXPECT_THROW(quantizer.DistanceFromLUT(lut.data(), nullptr), std::invalid_argument);
}

TEST(ScalarQuantizer, LUTAndTotalDistanceOverflowAreExplicit) {
    ann::ScalarQuantizer quantizer(2);
    const float training[] = {0, 0};
    quantizer.Train(1, training);
    const float query[] = {0, std::numeric_limits<float>::max()};
    std::vector<float> lut(quantizer.LutSize(), -7);
    EXPECT_THROW(quantizer.BuildLUT(query, lut.data()), std::overflow_error);
    EXPECT_EQ(lut.front(), -7);
    EXPECT_EQ(lut.back(), -7);
    std::fill(lut.begin(), lut.end(), std::numeric_limits<float>::max());
    const uint8_t code[] = {0, 0};
    EXPECT_THROW(quantizer.DistanceFromLUT(lut.data(), code), std::overflow_error);
    lut[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(quantizer.DistanceFromLUT(lut.data(), code), std::invalid_argument);
    lut[0] = -1;
    EXPECT_THROW(quantizer.DistanceFromLUT(lut.data(), code), std::invalid_argument);
}

TEST(ScalarQuantizer, RejectsInvalidDimensionsAndCountsModelCapacity) {
    EXPECT_THROW(ann::ScalarQuantizer(0), std::invalid_argument);
    EXPECT_THROW(ann::ScalarQuantizer(std::numeric_limits<size_t>::max()), std::length_error);
    ann::ScalarQuantizer quantizer(3);
    EXPECT_GE(quantizer.MemoryUsage(), sizeof(quantizer));
    const float training[] = {1, 2, 3};
    quantizer.Train(1, training);
    EXPECT_GE(quantizer.MemoryUsage(), sizeof(quantizer) + 6 * sizeof(float));
    EXPECT_EQ(quantizer.CodeSize(), 3U);
    EXPECT_EQ(quantizer.LutSize(), 768U);
}

}  // namespace
