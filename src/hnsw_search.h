// HNSW 内部搜索原语。图只读，临时状态由调用方独占。
#pragma once
#ifndef ANN_SRC_HNSW_SEARCH_H
#define ANN_SRC_HNSW_SEARCH_H

#include <span>
#include <utility>
#include <vector>

#include "hnsw_graph.h"

#include "ann/distance.h"

namespace ann::detail {

using HNSWCandidate = std::pair<dist_t, idx_t>;

// 不放在索引成员中；并发查询必须分别持有工作区。
struct HNSWSearchWorkspace {
    // 在搜索前预留最坏 N 个待扩展项和 min(ef,N) 个结果项。
    void Prepare(size_t nodes, size_t ef);
    size_t MemoryUsage() const noexcept;

    std::vector<HNSWCandidate> frontier;
    std::vector<HNSWCandidate> results;
    std::vector<uint8_t> visited;
};

// 仅严格距离改善才移动；等距不跨越，空图返回 kInvalidIdx。
idx_t HNSWGreedyDescend(const HNSWGraphStorage& graph,
                        std::span<const float> query,
                        idx_t entry,
                        int level,
                        DistFn distance);

// 返回最多 ef 个已发现候选，按距离、ID 排序；可达节点不足时可以更少。
// workspace 必须已 Prepare；函数不扩容，但每次会清零 N 字节访问标记。
// 结果视图由 workspace 持有，下一次搜索/准备/修改工作区可使其失效。
std::span<const HNSWCandidate> HNSWSearchLayer(const HNSWGraphStorage& graph,
                                               std::span<const float> query,
                                               idx_t entry,
                                               size_t ef,
                                               int level,
                                               DistFn distance,
                                               HNSWSearchWorkspace& workspace);

}  // namespace ann::detail

#endif  // ANN_SRC_HNSW_SEARCH_H
