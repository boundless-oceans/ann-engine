#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory_resource>
#include <string>
#include <type_traits>
#include <vector>

#include "ann/memory_pool.h"

namespace {

static_assert(!std::is_copy_constructible_v<ann::BuildArena>);
static_assert(!std::is_move_constructible_v<ann::BuildArena>);

TEST(BuildArena, EmptyResetAndStableResource) {
    ann::BuildArena arena;
    auto* resource = arena.Resource();
    ASSERT_NE(resource, nullptr);
    EXPECT_EQ(arena.UsedBytes(), 0u);
    EXPECT_EQ(arena.CapacityBytes(), 0u);
    arena.Reset();
    arena.Reset();
    EXPECT_EQ(arena.Resource(), resource);
    EXPECT_EQ(arena.UsedBytes(), 0u);
    EXPECT_EQ(arena.CapacityBytes(), 0u);
}

TEST(BuildArena, CountsPayloadWithoutReclaimingIndividualAllocations) {
    ann::BuildArena arena(128);
    auto* resource = arena.Resource();
    void* first = resource->allocate(3, 1);
    void* second = resource->allocate(17, 64);
    EXPECT_EQ(arena.UsedBytes(), 20u);
    EXPECT_GE(arena.CapacityBytes(), arena.UsedBytes());
    const auto capacity = arena.CapacityBytes();
    resource->deallocate(first, 3, 1);
    resource->deallocate(second, 17, 64);
    EXPECT_EQ(arena.UsedBytes(), 20u);
    EXPECT_EQ(arena.CapacityBytes(), capacity);
    arena.Reset();
    EXPECT_EQ(arena.UsedBytes(), 0u);
    EXPECT_EQ(arena.CapacityBytes(), capacity);
}

TEST(BuildArena, MixedAlignmentAndMultipleBlocksPreserveContents) {
    ann::BuildArena arena(64);
    const std::array<size_t, 6> sizes{7, 513, 2049, 13, 10001, 31};
    const std::array<size_t, 6> alignments{1, 32, 128, 8, 4096, 64};
    std::array<void*, 6> allocations{};
    size_t total = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto* p = arena.Resource()->allocate(sizes[i], alignments[i]);
        allocations[i] = p;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % alignments[i], 0u);
        std::memset(p, static_cast<int>(i + 1), sizes[i]);
        total += sizes[i];
    }
    EXPECT_EQ(arena.UsedBytes(), total);
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto* data = static_cast<const unsigned char*>(allocations[i]);
        for (size_t j = 0; j < sizes[i]; ++j) {
            ASSERT_EQ(data[j], i + 1);
        }
    }
}

TEST(BuildArena, RepeatedWorkloadReusesCapacityAcrossReset) {
    ann::BuildArena arena(64);
    size_t capacity = 0;
    for (int round = 0; round < 5; ++round) {
        size_t payload = 0;
        for (size_t i = 0; i < 100; ++i) {
            const size_t size = 17 + i * 3;
            auto* p = arena.Resource()->allocate(size, 64);
            std::memset(p, round, size);
            payload += size;
        }
        EXPECT_EQ(arena.UsedBytes(), payload);
        if (round == 0) {
            capacity = arena.CapacityBytes();
        } else {
            EXPECT_EQ(arena.CapacityBytes(), capacity);
        }
        arena.Reset();
        EXPECT_EQ(arena.UsedBytes(), 0u);
        EXPECT_EQ(arena.CapacityBytes(), capacity);
    }
}

TEST(BuildArena, LaterLargerWorkloadCanGrowAndBeReused) {
    ann::BuildArena arena(64);
    auto* resource = arena.Resource();
    auto* first = resource->allocate(8, 8);
    const auto small_capacity = arena.CapacityBytes();
    resource->deallocate(first, 8, 8);
    arena.Reset();
    auto* large = resource->allocate(100000, 128);
    const auto large_capacity = arena.CapacityBytes();
    EXPECT_GT(large_capacity, small_capacity);
    resource->deallocate(large, 100000, 128);
    arena.Reset();
    auto* reused = resource->allocate(100000, 128);
    EXPECT_EQ(arena.CapacityBytes(), large_capacity);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(reused) % 128, 0u);
    resource->deallocate(reused, 100000, 128);
}

TEST(BuildArena, PmrContainersReleaseBeforeResetAndRebuild) {
    ann::BuildArena arena(128);
    for (int round = 0; round < 2; ++round) {
        {
            std::pmr::vector<std::pmr::string> values(arena.Resource());
            for (size_t i = 0; i < 20; ++i) {
                values.emplace_back(100 + i, static_cast<char>('a' + round));
            }
            for (size_t i = 0; i < values.size(); ++i) {
                EXPECT_EQ(values[i].size(), 100 + i);
                EXPECT_EQ(values[i].get_allocator().resource(), arena.Resource());
                EXPECT_EQ(values[i].front(), static_cast<char>('a' + round));
            }
            EXPECT_GT(arena.UsedBytes(), 0u);
        }
        // clear 不够：容器可能仍持有 capacity，必须先结束生命周期再重置。
        arena.Reset();
        EXPECT_EQ(arena.UsedBytes(), 0u);
    }
}

struct Lifetime {
    static inline int alive = 0;
    Lifetime() { ++alive; }
    Lifetime(const Lifetime&) { ++alive; }
    Lifetime& operator=(const Lifetime&) = default;
    ~Lifetime() { --alive; }
};

TEST(BuildArena, ObjectsAreDestroyedByTheirContainers) {
    ann::BuildArena arena;
    {
        std::pmr::vector<Lifetime> values(arena.Resource());
        values.resize(100);
        EXPECT_EQ(Lifetime::alive, 100);
    }
    EXPECT_EQ(Lifetime::alive, 0);
    EXPECT_GT(arena.UsedBytes(), 0u);
    arena.Reset();
    EXPECT_EQ(arena.UsedBytes(), 0u);
}

TEST(BuildArena, ZeroByteAllocationAndZeroInitialHint) {
    ann::BuildArena arena(0);
    auto* p = arena.Resource()->allocate(0, 128);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 128, 0u);
    EXPECT_EQ(arena.UsedBytes(), 0u);
    EXPECT_GT(arena.CapacityBytes(), 0u);
    arena.Resource()->deallocate(p, 0, 128);
    arena.Reset();
}

TEST(BuildArena, PayloadOverflowDoesNotCorruptExistingAllocation) {
    ann::BuildArena arena(128);
    auto* p = static_cast<unsigned char*>(arena.Resource()->allocate(16, 8));
    p[0] = 42;
    const auto capacity = arena.CapacityBytes();
    EXPECT_THROW(
        static_cast<void>(arena.Resource()->allocate(std::numeric_limits<size_t>::max(), 8)),
        std::bad_alloc);
    EXPECT_EQ(arena.UsedBytes(), 16u);
    EXPECT_EQ(arena.CapacityBytes(), capacity);
    EXPECT_EQ(p[0], 42);
    arena.Resource()->deallocate(p, 16, 8);
}

TEST(BuildArena, DifferentArenasHaveDistinctResourceIdentity) {
    ann::BuildArena first;
    ann::BuildArena second;
    EXPECT_TRUE(first.Resource()->is_equal(*first.Resource()));
    EXPECT_FALSE(first.Resource()->is_equal(*second.Resource()));
}

}  // namespace
