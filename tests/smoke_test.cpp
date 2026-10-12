// 骨架连通性测试：证明「配置 → 构建 → 运行测试」这条链路是通的。
//
// 这里刻意不依赖任何算法实现（distance / flat），
// 因此即使 src/ 下的实现尚未完成，本文件也能编译、链接并通过。
// 算法测试另见 flat_test.cpp / distance_test.cpp。

#include <gtest/gtest.h>

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <vector>

#include "ann/types.h"

namespace {

// idx_t 必须是无符号 32 位：它同时充当数组下标与持久化格式的一部分，
// 宽度变化会破坏索引文件的兼容性。
TEST(Types, IndexTypeIsUint32) {
    static_assert(std::is_same_v<ann::idx_t, std::uint32_t>);
    EXPECT_EQ(sizeof(ann::idx_t), 4u);
}

// 哨兵值必须是该类型能表示的最大值，且与任何合法下标都不冲突。
TEST(Types, InvalidIdxIsMaxSentinel) {
    EXPECT_EQ(ann::kInvalidIdx, std::numeric_limits<ann::idx_t>::max());
    EXPECT_GT(ann::kInvalidIdx, 0u);
}

// 对齐常量需覆盖 AVX2 的 32 字节要求。
TEST(Types, AlignmentCoversAvx2) {
    EXPECT_GE(ann::kAlignment, 32u);
    EXPECT_EQ(ann::kAlignment % 32u, 0u);
}

// Metric 的取值是持久化格式的一部分，不得随意调整。
TEST(Types, MetricEncodingIsStable) {
    EXPECT_EQ(static_cast<std::uint8_t>(ann::Metric::L2), 0u);
    EXPECT_EQ(static_cast<std::uint8_t>(ann::Metric::InnerProduct), 1u);
    EXPECT_EQ(static_cast<std::uint8_t>(ann::Metric::Cosine), 2u);
}

// ResultView 是纯出参结构，默认构造应为全空，
// 便于调用方在栈上零初始化后直接传入。
TEST(Types, ResultViewDefaultsToNull) {
    ann::ResultView r;
    EXPECT_EQ(r.dists, nullptr);
    EXPECT_EQ(r.ids, nullptr);
    EXPECT_EQ(r.k, 0u);
}

// 编译期确认 C++20 可用的几个关键设施确实存在（而非仅在标准里）。
TEST(Standard, Cxx20FacilitiesAvailable) {
    // concepts
    static_assert(std::integral<int>);
    // std::span
    int buf[4] = {1, 2, 3, 4};
    std::span<int> s{buf};
    EXPECT_EQ(s.size(), 4u);
    // std::pmr
    std::pmr::monotonic_buffer_resource pool;
    std::pmr::vector<int> v{&pool};
    v.push_back(42);
    EXPECT_EQ(v[0], 42);
    // std::bit
    EXPECT_EQ(std::bit_cast<std::uint32_t>(1.0f), 0x3F800000u);
}

}  // namespace
