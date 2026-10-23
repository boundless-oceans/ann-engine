// HNSW 内部选邻规则；候选按中心距离/ID 升序且 ID 唯一，由调用方提供输出空间。
#pragma once
#ifndef ANN_SRC_HNSW_NEIGHBORS_H
#define ANN_SRC_HNSW_NEIGHBORS_H

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

#include "hnsw_search.h"

namespace ann::detail {

// between(a,b) 按 ID 计算候选间距离，可由调用方解析尚未 Append 的新点。
// 容量足够时保留全部；超限时不回填已拒绝候选，因此允许少于输出容量。
// output 不得与 candidates 重叠；异常可能写入部分 output，但不修改输入/图。
template<class PairDistance>
size_t HNSWSelectNeighbors(std::span<const HNSWCandidate> candidates,
                           std::span<HNSWCandidate> output,
                           PairDistance&& between) {
    if (output.empty()) {
        return 0;
    }
    for (const auto& candidate : candidates) {
        if (!std::isfinite(candidate.first)) {
            throw std::invalid_argument("neighbor selection requires finite distances");
        }
    }
    if (candidates.size() <= output.size()) {
        std::copy(candidates.begin(), candidates.end(), output.begin());
        return candidates.size();
    }
    size_t count = 0;
    for (const auto& candidate : candidates) {
        bool covered = false;
        for (size_t i = 0; i < count; ++i) {
            const auto distance = between(candidate.second, output[i].second);
            if (!std::isfinite(distance)) {
                throw std::invalid_argument("neighbor selection requires finite pair distances");
            }
            // 使用严格小于；等距本身不会导致拒绝，避免把平局当成重复覆盖。
            if (distance < candidate.first) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            output[count++] = candidate;
            if (count == output.size()) {
                break;
            }
        }
    }
    return count;
}

}  // namespace ann::detail

#endif  // ANN_SRC_HNSW_NEIGHBORS_H
