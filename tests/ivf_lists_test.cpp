#include "ivf_lists.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

TEST(IVFLists, HandComputedStableOriginalIds) {
    const std::vector<uint32_t> assignments{2, 0, 2, 1, 0, 2};
    const auto before = assignments;
    const auto lists = ann::detail::BuildIVFLists(assignments, 3);
    EXPECT_EQ(lists.offsets, (std::vector<uint32_t>{0, 2, 3, 6}));
    EXPECT_EQ(lists.ids, (std::vector<uint32_t>{1, 4, 3, 0, 2, 5}));
    EXPECT_EQ(assignments, before);
}

TEST(IVFLists, EmptyListsUseRepeatedOffsets) {
    const uint32_t assignments[] = {1, 1, 3};
    const auto lists = ann::detail::BuildIVFLists(assignments, 5);
    EXPECT_EQ(lists.offsets, (std::vector<uint32_t>{0, 0, 2, 2, 3, 3}));
    EXPECT_EQ(lists.ids, (std::vector<uint32_t>{0, 1, 2}));
}

TEST(IVFLists, EmptyInputHasValidOffsets) {
    const auto lists = ann::detail::BuildIVFLists({}, 3);
    EXPECT_EQ(lists.offsets, (std::vector<uint32_t>{0, 0, 0, 0}));
    EXPECT_TRUE(lists.ids.empty());
}

TEST(IVFLists, EveryIdAppearsOnceInItsAssignedList) {
    std::vector<uint32_t> assignments(1000);
    for (size_t i = 0; i < assignments.size(); ++i) {
        assignments[i] = static_cast<uint32_t>((i * 17) % 13);
    }
    const auto lists = ann::detail::BuildIVFLists(assignments, 13);
    ASSERT_EQ(lists.offsets.back(), assignments.size());
    for (size_t c = 0; c < 13; ++c) {
        EXPECT_LE(lists.offsets[c], lists.offsets[c + 1]);
        for (size_t pos = lists.offsets[c]; pos < lists.offsets[c + 1]; ++pos) {
            EXPECT_EQ(assignments[lists.ids[pos]], c);
            if (pos > lists.offsets[c]) {
                EXPECT_LT(lists.ids[pos - 1], lists.ids[pos]);
            }
        }
    }
    auto sorted = lists.ids;
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < sorted.size(); ++i) {
        EXPECT_EQ(sorted[i], i);
    }
}

TEST(IVFLists, RejectsInvalidCountsAndClusterIds) {
    EXPECT_THROW(ann::detail::BuildIVFLists({}, 0), std::invalid_argument);
    const uint32_t invalid[] = {0, 3};
    EXPECT_THROW(ann::detail::BuildIVFLists(invalid, 3), std::invalid_argument);
    EXPECT_THROW(ann::detail::BuildIVFLists({}, static_cast<size_t>(UINT32_MAX) + 1),
                 std::invalid_argument);
}

}  // namespace
