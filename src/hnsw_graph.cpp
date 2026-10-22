#include "hnsw_graph.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ann::detail {
namespace {

size_t CheckedGrowth(size_t old_size, size_t count, size_t stride) {
    if (count > (std::numeric_limits<size_t>::max() - old_size) / stride) {
        throw std::length_error("graph storage size overflow");
    }
    return old_size + count * stride;
}

}  // namespace

HNSWGraphStorage::HNSWGraphStorage(size_t dim, size_t upper_slots, size_t base_slots)
    : dim_(dim), upper_slots_(upper_slots), base_slots_(base_slots) {
    if (dim == 0 || upper_slots == 0 || base_slots == 0) {
        throw std::invalid_argument("dimension and neighbor capacities must be positive");
    }
    if (dim > std::numeric_limits<size_t>::max() / sizeof(float) || upper_slots > kInvalidIdx ||
        base_slots > kInvalidIdx) {
        throw std::length_error("graph storage parameters exceed addressable range");
    }
}

idx_t HNSWGraphStorage::Append(std::span<const float> vector, uint8_t level) {
    if (vector.size() != dim_) {
        throw std::invalid_argument("vector dimension does not match graph dimension");
    }
    if (Size() >= kInvalidIdx) {
        throw std::length_error("graph node count exceeds available IDs");
    }
    const size_t old_n = Size();
    const size_t old_vectors = vectors_.Size();
    const size_t old_l0 = links_l0_.size();
    const size_t old_upper = upper_pool_.size();
    const size_t next_vectors = CheckedGrowth(old_vectors, 1, dim_);
    const size_t next_l0 = CheckedGrowth(old_l0, 1, base_slots_);
    const size_t next_upper = CheckedGrowth(old_upper, level, upper_slots_);
    // 上层池中可持久化的引用是 uint32_t 槽位偏移，不能截断成无效地址。
    if (next_upper > kInvalidIdx) {
        throw std::length_error("upper layer pool exceeds 32-bit offsets");
    }

    // 支持使用已有向量追加副本：扩容前保存偏移，扩容后重新取得数据地址。
    const auto begin = reinterpret_cast<uintptr_t>(vectors_.Data());
    const auto source = reinterpret_cast<uintptr_t>(vector.data());
    const bool aliases =
        old_vectors != 0 && source >= begin && (source - begin) / sizeof(float) < old_vectors;
    const size_t source_offset = aliases ? (source - begin) / sizeof(float) : 0;
    if (aliases && dim_ > old_vectors - source_offset) {
        throw std::invalid_argument("source range exceeds stored vectors");
    }

    try {
        vectors_.Resize(next_vectors);
        links_l0_.resize(next_l0, kInvalidIdx);
        upper_pool_.resize(next_upper, kInvalidIdx);
        upper_offset_.push_back(level == 0 ? kInvalidIdx : static_cast<uint32_t>(old_upper));
        node_level_.push_back(level);
    } catch (...) {
        // 回滚逻辑长度与内容；成功过的扩容可能保留容量并使旧视图失效。
        vectors_.Resize(old_vectors);
        links_l0_.resize(old_l0);
        upper_pool_.resize(old_upper);
        upper_offset_.resize(old_n);
        node_level_.resize(old_n);
        throw;
    }
    const float* source_data = aliases ? vectors_.Data() + source_offset : vector.data();
    std::copy_n(source_data, dim_, vectors_.Data() + old_vectors);
    const auto id = static_cast<idx_t>(old_n);
    // 新最高层节点成为入口；同层追加保持原入口，便于后续确定性构建。
    if (static_cast<int>(level) > max_level_) {
        entry_point_ = id;
        max_level_ = level;
    }
    return id;
}

void HNSWGraphStorage::CheckNode(idx_t node) const {
    if (node >= Size()) {
        throw std::out_of_range("graph node does not exist");
    }
}

size_t HNSWGraphStorage::LinkOffset(idx_t node, uint8_t level) const {
    CheckNode(node);
    if (level > node_level_[node]) {
        throw std::out_of_range("node does not belong to requested layer");
    }
    return level == 0 ? static_cast<size_t>(node) * base_slots_
                      : static_cast<size_t>(upper_offset_[node]) +
                            static_cast<size_t>(level - 1) * upper_slots_;
}

const idx_t* HNSWGraphStorage::Links(idx_t node, uint8_t level) const {
    const size_t offset = LinkOffset(node, level);
    return (level == 0 ? links_l0_.data() : upper_pool_.data()) + offset;
}

idx_t* HNSWGraphStorage::Links(idx_t node, uint8_t level) {
    const size_t offset = LinkOffset(node, level);
    return (level == 0 ? links_l0_.data() : upper_pool_.data()) + offset;
}

void HNSWGraphStorage::SetNeighbors(idx_t node, uint8_t level, std::span<const idx_t> neighbors) {
    idx_t* slots = Links(node, level);
    if (neighbors.size() > SlotCount(level)) {
        throw std::length_error("neighbor list exceeds layer capacity");
    }
    // 一次性验证后再修改；M 很小，直接两两去重，不引入临时哈希表。
    for (size_t i = 0; i < neighbors.size(); ++i) {
        const idx_t neighbor = neighbors[i];
        CheckNode(neighbor);
        if (neighbor == node || node_level_[neighbor] < level) {
            throw std::invalid_argument("invalid self edge or neighbor layer");
        }
        for (size_t j = 0; j < i; ++j) {
            if (neighbors[j] == neighbor) {
                throw std::invalid_argument("duplicate neighbor ID");
            }
        }
    }
    if (!neighbors.empty()) {
        // 调用方可能直接传入现有邻接视图，允许源与目标区间重叠。
        std::memmove(slots, neighbors.data(), neighbors.size_bytes());
    }
    std::fill(slots + neighbors.size(), slots + SlotCount(level), kInvalidIdx);
}

std::span<const float> HNSWGraphStorage::Vector(idx_t node) const {
    CheckNode(node);
    return {vectors_.Data() + static_cast<size_t>(node) * dim_, dim_};
}

std::span<const idx_t> HNSWGraphStorage::Neighbors(idx_t node, uint8_t level) const {
    const idx_t* first = Links(node, level);
    const idx_t* end = std::find(first, first + SlotCount(level), kInvalidIdx);
    return {first, static_cast<size_t>(end - first)};
}

uint8_t HNSWGraphStorage::Level(idx_t node) const {
    CheckNode(node);
    return node_level_[node];
}

size_t HNSWGraphStorage::SlotCount(uint8_t level) const noexcept {
    return level == 0 ? base_slots_ : upper_slots_;
}

size_t HNSWGraphStorage::MemoryUsage() const noexcept {
    return sizeof(*this) + vectors_.CapacityBytes() + links_l0_.capacity() * sizeof(idx_t) +
           upper_pool_.capacity() * sizeof(idx_t) + upper_offset_.capacity() * sizeof(uint32_t) +
           node_level_.capacity() * sizeof(uint8_t);
}

}  // namespace ann::detail
