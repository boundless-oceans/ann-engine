#include "ann/memory_pool.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "ann/flat.h"

namespace {

TEST(AlignedAllocator, ArbitraryByteCountsAndAlignments) {
    for (const size_t align : {32u, 64u, 128u}) {
        for (const size_t bytes : {1u, 3u, 31u, 32u, 33u, 513u}) {
            void* p = ann::AlignedAllocator::Allocate(bytes, align);
            ASSERT_NE(p, nullptr);
            EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % align, 0u);
            auto* data = static_cast<unsigned char*>(p);
            // 首尾都可读写，特别覆盖非对齐整数倍的申请长度。
            data[0] = 17;
            data[bytes - 1] = 29;
            EXPECT_EQ(data[bytes - 1], 29);
            ann::AlignedAllocator::Deallocate(p);
        }
    }
}

TEST(AlignedAllocator, EmptyAllocationAndNullDeallocation) {
    EXPECT_EQ(ann::AlignedAllocator::Allocate(0), nullptr);
    ann::AlignedAllocator::Deallocate(nullptr);
}

TEST(AlignedAllocator, RejectsInvalidAlignment) {
    for (const size_t align : {0u, 1u, 3u, 12u, 48u}) {
        EXPECT_THROW(ann::AlignedAllocator::Allocate(1, align), std::invalid_argument);
        EXPECT_THROW(ann::AlignedAllocator::Allocate(0, align), std::invalid_argument);
    }
}

TEST(AlignedVector, EmptyAndValueInitializedGrowth) {
    ann::AlignedVector<float> values;
    EXPECT_EQ(values.Size(), 0u);
    EXPECT_EQ(values.CapacityBytes(), 0u);
    values.Resize(33);
    EXPECT_EQ(values.Size(), 33u);
    EXPECT_GE(values.CapacityBytes(), 33 * sizeof(float));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(values.Data()) % ann::kAlignment, 0u);
    for (size_t i = 0; i < values.Size(); ++i) {
        EXPECT_EQ(values.Data()[i], 0.0f);
    }
}

TEST(AlignedVector, GrowthAndShrinkPreserveValuesAndAlignment) {
    ann::AlignedVector<int> values(64);
    values.Resize(17);
    for (size_t i = 0; i < 17; ++i) {
        values.Data()[i] = static_cast<int>(i * 3);
    }
    values.Resize(1001);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(values.Data()) % 64, 0u);
    for (size_t i = 0; i < 17; ++i) {
        EXPECT_EQ(values.Data()[i], static_cast<int>(i * 3));
    }
    const auto capacity = values.CapacityBytes();
    values.Resize(5);
    values.Resize(17);
    EXPECT_EQ(values.CapacityBytes(), capacity);
    EXPECT_EQ(values.Data()[4], 12);
    EXPECT_EQ(values.Data()[5], 0);
}

TEST(AlignedVector, CopyAndAssignmentOwnIndependentStorage) {
    ann::AlignedVector<int> original(64);
    original.Resize(3);
    original.Data()[0] = 42;
    auto copy = original;
    ann::AlignedVector<int> assigned;
    assigned = original;
    EXPECT_NE(copy.Data(), original.Data());
    EXPECT_NE(assigned.Data(), original.Data());
    EXPECT_EQ(copy.Data()[0], 42);
    EXPECT_EQ(assigned.Data()[0], 42);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(assigned.Data()) % 64, 0u);
    copy.Data()[0] = 7;
    EXPECT_EQ(original.Data()[0], 42);
}

TEST(AlignedVector, MoveTransfersStorageAndAllocator) {
    ann::AlignedVector<int> original(128);
    original.Resize(3);
    original.Data()[0] = 23;
    auto* storage = original.Data();
    auto moved = std::move(original);
    EXPECT_EQ(moved.Data(), storage);
    ann::AlignedVector<int> assigned;
    assigned = std::move(moved);
    EXPECT_EQ(assigned.Data(), storage);
    assigned.Resize(100);
    EXPECT_EQ(assigned.Data()[0], 23);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(assigned.Data()) % 128, 0u);
}

struct alignas(128) Tracked {
    static inline int alive = 0;
    Tracked() { ++alive; }
    Tracked(const Tracked&) { ++alive; }
    Tracked& operator=(const Tracked&) = default;
    ~Tracked() { --alive; }
};

TEST(AlignedVector, OveralignedObjectsHaveBalancedLifetimes) {
    EXPECT_EQ(Tracked::alive, 0);
    {
        ann::AlignedVector<Tracked> values;
        values.Resize(3);
        EXPECT_EQ(Tracked::alive, 3);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(values.Data()) % alignof(Tracked), 0u);
        values.Resize(30);
        EXPECT_EQ(Tracked::alive, 30);
        values.Resize(1);
        EXPECT_EQ(Tracked::alive, 1);
    }
    EXPECT_EQ(Tracked::alive, 0);
}

TEST(AlignedVector, ImpossibleGrowthLeavesExistingStorageIntact) {
    ann::AlignedVector<float> values;
    values.Resize(1);
    values.Data()[0] = 17.0f;
    auto* storage = values.Data();
    EXPECT_THROW(values.Resize(std::numeric_limits<size_t>::max()), std::length_error);
    EXPECT_EQ(values.Size(), 1u);
    EXPECT_EQ(values.Data(), storage);
    EXPECT_EQ(values.Data()[0], 17.0f);
}

TEST(AlignedVector, InvalidAlignmentIsRejectedBeforeAllocation) {
    EXPECT_THROW(ann::AlignedVector<float> values(24), std::invalid_argument);
}

TEST(FlatAlignedStorage, RepeatedAddsPreserveContiguousRowsAndResults) {
    for (const size_t dim : {1u, 7u, 8u, 127u, 128u, 129u}) {
        SCOPED_TRACE(dim);
        ann::FlatIndex index(dim, ann::Metric::L2);
        std::vector<float> row(dim);
        for (size_t i = 0; i < 13; ++i) {
            std::fill(row.begin(), row.end(), static_cast<float>(i));
            index.Add(1, row.data());
            EXPECT_EQ(reinterpret_cast<uintptr_t>(index.Data()) % ann::kAlignment, 0u);
        }
        EXPECT_EQ(index.Size(), 13u);
        for (size_t i = 0; i < 13; ++i) {
            EXPECT_EQ(index.Data()[i * dim], static_cast<float>(i));
        }
        std::fill(row.begin(), row.end(), 6.0f);
        float dist = -1.0f;
        ann::idx_t id = ann::kInvalidIdx;
        index.Search(row.data(), 1, {&dist, &id, 1});
        EXPECT_EQ(id, 6u);
        EXPECT_EQ(dist, 0.0f);
        EXPECT_GE(index.MemoryUsage(), sizeof(index) + 13 * dim * sizeof(float));
    }
}

TEST(FlatAlignedStorage, AddFromOwnStorageSurvivesReallocation) {
    ann::FlatIndex index(2, ann::Metric::L2);
    const float base[] = {1, 2, 3, 4};
    index.Add(2, base);
    index.Add(2, index.Data());
    EXPECT_EQ(index.Size(), 4u);
    EXPECT_EQ(index.Data()[4], 1.0f);
    EXPECT_EQ(index.Data()[7], 4.0f);
    EXPECT_THROW(index.Add(2, index.Data() + 6), std::invalid_argument);
    EXPECT_EQ(index.Size(), 4u);
}

TEST(FlatAlignedStorage, RejectsOverflowBeforeReadingInput) {
    const float value = 1.0f;
    ann::FlatIndex index(2, ann::Metric::L2);
    EXPECT_THROW(index.Add(std::numeric_limits<size_t>::max(), &value), std::length_error);
    EXPECT_EQ(index.Size(), 0u);
    ann::FlatIndex huge(std::numeric_limits<size_t>::max(), ann::Metric::L2);
    EXPECT_THROW(huge.Add(2, &value), std::length_error);
    EXPECT_EQ(huge.Size(), 0u);
}

}  // namespace
