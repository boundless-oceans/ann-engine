#include "hnsw_search.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace ann::detail {
namespace {

uint8_t ValidateSearch(const HNSWGraphStorage& graph,
                       std::span<const float> query,
                       int level,
                       DistFn distance) {
    if (query.size() != graph.Dim() || distance == nullptr) {
        throw std::invalid_argument("query dimension or distance function is invalid");
    }
    if (level < 0 || level > std::numeric_limits<uint8_t>::max()) {
        throw std::out_of_range("layer is outside graph storage range");
    }
    return static_cast<uint8_t>(level);
}

dist_t DistanceTo(const HNSWGraphStorage& graph,
                  std::span<const float> query,
                  idx_t node,
                  DistFn distance) {
    const dist_t value = distance(query.data(), graph.Vector(node).data(), graph.Dim());
    // 保持有限距离契约，拒绝 NaN 和溢出后的 Inf 进入候选排序。
    if (!std::isfinite(value)) {
        throw std::invalid_argument("graph search requires finite distances");
    }
    return value;
}

void ValidateEntry(const HNSWGraphStorage& graph, idx_t entry, uint8_t level) {
    if (graph.Level(entry) < level) {
        throw std::out_of_range("entry node does not exist in requested layer");
    }
}

}  // namespace

void HNSWSearchWorkspace::Prepare(size_t nodes, size_t ef) {
    if (nodes > kInvalidIdx) {
        throw std::length_error("workspace node count exceeds graph ID range");
    }
    frontier.clear();
    results.clear();
    frontier.reserve(nodes);
    results.reserve(std::min(ef, nodes));
    visited.resize(nodes);
}

size_t HNSWSearchWorkspace::MemoryUsage() const noexcept {
    return sizeof(*this) + frontier.capacity() * sizeof(HNSWCandidate) +
           results.capacity() * sizeof(HNSWCandidate) + visited.capacity() * sizeof(uint8_t);
}

idx_t HNSWGreedyDescend(const HNSWGraphStorage& graph,
                        std::span<const float> query,
                        idx_t entry,
                        int level,
                        DistFn distance) {
    const auto layer = ValidateSearch(graph, query, level, distance);
    if (graph.Size() == 0) {
        return kInvalidIdx;
    }
    ValidateEntry(graph, entry, layer);
    idx_t current = entry;
    dist_t current_distance = DistanceTo(graph, query, current, distance);
    for (;;) {
        idx_t best = current;
        dist_t best_distance = current_distance;
        for (const auto neighbor : graph.Neighbors(current, layer)) {
            const auto value = DistanceTo(graph, query, neighbor, distance);
            if (value < best_distance) {
                best = neighbor;
                best_distance = value;
            }
        }
        if (best == current) {
            return current;
        }
        current = best;
        current_distance = best_distance;
    }
}

std::span<const HNSWCandidate> HNSWSearchLayer(const HNSWGraphStorage& graph,
                                               std::span<const float> query,
                                               idx_t entry,
                                               size_t ef,
                                               int level,
                                               DistFn distance,
                                               HNSWSearchWorkspace& workspace) {
    const auto layer = ValidateSearch(graph, query, level, distance);
    if (ef == 0 || graph.Size() == 0) {
        workspace.frontier.clear();
        workspace.results.clear();
        return workspace.results;
    }
    ValidateEntry(graph, entry, layer);
    const size_t limit = std::min(ef, graph.Size());
    if (workspace.visited.size() < graph.Size() || workspace.frontier.capacity() < graph.Size() ||
        workspace.results.capacity() < limit) {
        throw std::length_error("search workspace must be prepared for graph size and ef");
    }
    workspace.frontier.clear();
    workspace.results.clear();
    std::fill_n(workspace.visited.begin(), graph.Size(), uint8_t{0});
    const HNSWCandidate initial{DistanceTo(graph, query, entry, distance), entry};
    workspace.frontier.push_back(initial);
    workspace.results.push_back(initial);
    workspace.visited[entry] = 1;
    const std::greater<HNSWCandidate> closest_first;

    while (!workspace.frontier.empty()) {
        // frontier 是最小堆，优先扩展最近候选；results 是最大堆，暴露最差结果。
        std::pop_heap(workspace.frontier.begin(), workspace.frontier.end(), closest_first);
        const auto current = workspace.frontier.back();
        workspace.frontier.pop_back();
        if (workspace.results.size() == limit && current.first > workspace.results.front().first) {
            break;
        }
        for (const auto neighbor : graph.Neighbors(current.second, layer)) {
            if (workspace.visited[neighbor] != 0) {
                continue;
            }
            workspace.visited[neighbor] = 1;
            const HNSWCandidate candidate{DistanceTo(graph, query, neighbor, distance), neighbor};
            if (workspace.results.size() < limit || candidate < workspace.results.front()) {
                workspace.frontier.push_back(candidate);
                std::push_heap(workspace.frontier.begin(), workspace.frontier.end(), closest_first);
                if (workspace.results.size() == limit) {
                    std::pop_heap(workspace.results.begin(), workspace.results.end());
                    workspace.results.pop_back();
                }
                workspace.results.push_back(candidate);
                std::push_heap(workspace.results.begin(), workspace.results.end());
            }
        }
    }
    std::sort_heap(workspace.results.begin(), workspace.results.end());
    return workspace.results;
}

}  // namespace ann::detail
