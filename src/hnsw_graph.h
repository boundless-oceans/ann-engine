// HNSW 内部图存储：仅负责数据与边，不执行选邻、搜索或随机层数生成。
#pragma once
#ifndef ANN_SRC_HNSW_GRAPH_H
#define ANN_SRC_HNSW_GRAPH_H

#include <span>
#include <vector>

#include "ann/memory_pool.h"
#include "ann/types.h"

namespace ann::detail {

class HNSWGraphStorage {
public:
    HNSWGraphStorage(size_t dim, size_t upper_slots, size_t base_slots);

    // 显式给层数以便独立验证存储；返回追加后的稳定节点 ID。
    idx_t Append(std::span<const float> vector, uint8_t level);
    // 只替换一条有向邻接表；双向连接和超限剪枝由后续算法负责。
    // 拒绝无效层、越界 ID、自环、重复邻居与超容量列表，失败不改原邻接表。
    void SetNeighbors(idx_t node, uint8_t level, std::span<const idx_t> neighbors);

    std::span<const float> Vector(idx_t node) const;
    std::span<const idx_t> Neighbors(idx_t node, uint8_t level) const;
    uint8_t Level(idx_t node) const;
    size_t SlotCount(uint8_t level) const noexcept;
    size_t Size() const noexcept { return node_level_.size(); }
    size_t Dim() const noexcept { return dim_; }
    idx_t EntryPoint() const noexcept { return entry_point_; }
    int MaxLevel() const noexcept { return max_level_; }
    size_t MemoryUsage() const noexcept;

    // 返回的只读 span 不是快照；追加扩容可使其失效，修改边也会改变其内容。

private:
    void CheckNode(idx_t node) const;
    size_t LinkOffset(idx_t node, uint8_t level) const;
    const idx_t* Links(idx_t node, uint8_t level) const;
    idx_t* Links(idx_t node, uint8_t level);

    size_t dim_;
    size_t upper_slots_;
    size_t base_slots_;
    AlignedVector<float> vectors_;
    std::vector<idx_t> links_l0_;
    std::vector<idx_t> upper_pool_;
    std::vector<uint32_t> upper_offset_;
    std::vector<uint8_t> node_level_;
    idx_t entry_point_ = kInvalidIdx;
    int max_level_ = -1;
};

}  // namespace ann::detail

#endif  // ANN_SRC_HNSW_GRAPH_H
