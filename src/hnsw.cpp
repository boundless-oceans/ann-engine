#include "ann/hnsw.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory_resource>
#include <stdexcept>

#include "hnsw_graph.h"
#include "hnsw_level.h"
#include "hnsw_search.h"

namespace ann {
namespace {

// 预先保存所有改边方案，Append 成功后只进行不分配的邻接表替换。
struct LinkUpdate {
    idx_t node;
    uint8_t level;
    size_t offset;
    size_t count;
};

struct ArenaScope {
    explicit ArenaScope(BuildArena& resource) : arena(resource) { arena.Reset(); }
    ~ArenaScope() { arena.Reset(); }
    BuildArena& arena;
};

// 一批 Add 内复用，整批结束后释放；不把构建工作区长期留在索引中。
struct BuildContext {
    detail::HNSWSearchWorkspace workspace;
    std::vector<detail::HNSWCandidate> reverse;
    BuildArena arena;
};

dist_t CheckedDistance(DistFn distance, const float* a, const float* b, size_t dim) {
    const auto value = distance(a, b, dim);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("HNSW requires finite distances");
    }
    return value;
}

}  // namespace

struct HNSWIndex::State {
    State(size_t dim, Metric metric_value, HNSWParams params)
        : p(params),
          graph(dim, p.M, p.Mmax0),
          levels(p.M, p.seed),
          distance(SelectDistFn(metric_value)),
          metric(metric_value),
          ef_search(p.ef_search) {
        if (p.Mmax0 < p.M || p.ef_construction == 0 || p.ef_search == 0) {
            throw std::invalid_argument("require Mmax0 >= M and positive ef values");
        }
        if (metric_value != Metric::L2 && metric_value != Metric::InnerProduct &&
            metric_value != Metric::Cosine) {
            throw std::invalid_argument("unsupported HNSW metric");
        }
    }

    void Insert(std::span<const float> query, BuildContext& build) {
        const auto old_entry = graph.EntryPoint();
        const auto old_max = graph.MaxLevel();
        const auto random_before = levels;
        const auto level = levels.Next();
        const auto id = static_cast<idx_t>(graph.Size());
        auto& build_workspace = build.workspace;
        auto& reverse_candidates = build.reverse;
        ArenaScope reset(build.arena);
        // 这些容器先于 reset 析构，Reset 不会释放仍存活容器的底层内存。
        std::pmr::vector<LinkUpdate> updates(build.arena.Resource());
        std::pmr::vector<idx_t> links(build.arena.Resource());
        try {
            auto stage =
                [&](idx_t node, uint8_t layer, std::span<const detail::HNSWCandidate> chosen) {
                    const size_t offset = links.size();
                    for (const auto& candidate : chosen) {
                        links.push_back(candidate.second);
                    }
                    updates.push_back({node, layer, offset, chosen.size()});
                };
            if (old_entry != kInvalidIdx) {
                auto entry = old_entry;
                for (int layer = old_max; layer > level; --layer) {
                    entry = detail::HNSWGreedyDescend(graph, query, entry, layer, distance);
                }
                for (int layer = std::min(old_max, static_cast<int>(level)); layer >= 0; --layer) {
                    const auto candidates =
                        detail::HNSWSearchLayer(graph,
                                                query,
                                                entry,
                                                std::max(p.ef_construction, p.M),
                                                layer,
                                                distance,
                                                build_workspace);
                    const size_t count = std::min(p.M, candidates.size());
                    const auto selected = candidates.first(count);
                    stage(id, static_cast<uint8_t>(layer), selected);
                    const size_t capacity = graph.SlotCount(static_cast<uint8_t>(layer));
                    for (const auto& candidate : selected) {
                        const auto neighbor = candidate.second;
                        const auto neighbor_vector = graph.Vector(neighbor);
                        reverse_candidates.clear();
                        for (const auto old :
                             graph.Neighbors(neighbor, static_cast<uint8_t>(layer))) {
                            reverse_candidates.emplace_back(
                                CheckedDistance(distance,
                                                neighbor_vector.data(),
                                                graph.Vector(old).data(),
                                                graph.Dim()),
                                old);
                        }
                        reverse_candidates.emplace_back(
                            CheckedDistance(
                                distance, neighbor_vector.data(), query.data(), graph.Dim()),
                            id);
                        std::sort(reverse_candidates.begin(), reverse_candidates.end());
                        const size_t retained = std::min(capacity, reverse_candidates.size());
                        stage(neighbor,
                              static_cast<uint8_t>(layer),
                              {reverse_candidates.data(), retained});
                    }
                    // 上一层最接近的候选成为下一层入口，选邻仍只是最近 M 个。
                    if (!candidates.empty()) {
                        entry = candidates.front().second;
                    }
                }
            }
            graph.Append(query, level);
        } catch (...) {
            // 当前点尚未改边；恢复层数状态便于失败后重试，已完成的批量前缀保留。
            levels = random_before;
            throw;
        }
        // 已按图层和容量生成合法方案；此阶段无分配，不执行候选计算。
        for (const auto& update : updates) {
            graph.SetNeighbors(
                update.node, update.level, {links.data() + update.offset, update.count});
        }
    }

    const HNSWParams p;
    detail::HNSWGraphStorage graph;
    detail::HNSWLevelGenerator levels;
    DistFn distance;
    Metric metric;
    std::atomic<size_t> ef_search;
};

HNSWIndex::HNSWIndex(size_t dim, Metric metric, HNSWParams params)
    : state_(std::make_unique<State>(dim, metric, params)) {}

HNSWIndex::~HNSWIndex() = default;

void HNSWIndex::Add(size_t n, const float* vecs) {
    if (n == 0 || vecs == nullptr) {
        return;
    }
    auto& state = *state_;
    if (n > static_cast<size_t>(kInvalidIdx) - state.graph.Size() ||
        n > std::numeric_limits<size_t>::max() / state.graph.Dim() / sizeof(float)) {
        throw std::length_error("HNSW vector count or byte size overflow");
    }
    const size_t elements = n * state.graph.Dim();
    for (size_t i = 0; i < elements; ++i) {
        if (!std::isfinite(vecs[i])) {
            throw std::invalid_argument("HNSW requires finite input vectors");
        }
    }
    // 为整批最终节点数准备一次；搜索只清除当前图实际节点的访问标记。
    const size_t total = state.graph.Size() + n;
    BuildContext build;
    build.workspace.Prepare(total, std::max(state.p.ef_construction, state.p.M));
    build.reverse.reserve(std::min(state.p.Mmax0, total) + 1);
    for (size_t i = 0; i < n; ++i) {
        state.Insert({vecs + i * state.graph.Dim(), state.graph.Dim()}, build);
    }
}

void HNSWIndex::Search(const float* query, size_t k, ResultView out) const {
    if (query == nullptr || out.dists == nullptr || out.ids == nullptr) {
        return;
    }
    const size_t want = std::min(k, out.k);
    if (want == 0) {
        return;
    }
    const State& state = *state_;
    size_t written = 0;
    if (state.graph.Size() != 0) {
        const std::span<const float> q{query, state.graph.Dim()};
        auto entry = state.graph.EntryPoint();
        for (int layer = state.graph.MaxLevel(); layer > 0; --layer) {
            entry = detail::HNSWGreedyDescend(state.graph, q, entry, layer, state.distance);
        }
        const size_t ef = std::max(want, state.ef_search.load(std::memory_order_relaxed));
        // 用户确认的正确性基线：工作区仅在函数内存活，不复用索引成员。
        detail::HNSWSearchWorkspace workspace;
        workspace.Prepare(state.graph.Size(), ef);
        const auto candidates =
            detail::HNSWSearchLayer(state.graph, q, entry, ef, 0, state.distance, workspace);
        written = std::min(want, candidates.size());
        for (size_t i = 0; i < written; ++i) {
            out.dists[i] = candidates[i].first;
            out.ids[i] = candidates[i].second;
        }
    }
    for (size_t i = written; i < want; ++i) {
        out.dists[i] = std::numeric_limits<dist_t>::max();
        out.ids[i] = kInvalidIdx;
    }
}

size_t HNSWIndex::Size() const {
    return state_->graph.Size();
}
size_t HNSWIndex::Dim() const {
    return state_->graph.Dim();
}
Metric HNSWIndex::GetMetric() const {
    return state_->metric;
}
size_t HNSWIndex::MemoryUsage() const {
    const State& state = *state_;
    return sizeof(*this) + sizeof(State) + state.graph.MemoryUsage() - sizeof(state.graph);
}
void HNSWIndex::SetEfSearch(size_t ef) noexcept {
    state_->ef_search.store(std::max(ef, size_t{1}), std::memory_order_relaxed);
}
size_t HNSWIndex::GetEfSearch() const noexcept {
    return state_->ef_search.load(std::memory_order_relaxed);
}
const HNSWParams& HNSWIndex::Params() const noexcept {
    return state_->p;
}

HNSWIndex::Stats HNSWIndex::GetStats() const {
    const auto& graph = state_->graph;
    Stats stats;
    stats.nodes = graph.Size();
    stats.max_level = graph.MaxLevel();
    for (idx_t node = 0; node < graph.Size(); ++node) {
        stats.edges_level0 += graph.Neighbors(node, 0).size();
        for (int layer = 1; layer <= graph.Level(node); ++layer) {
            stats.edges_upper += graph.Neighbors(node, static_cast<uint8_t>(layer)).size();
        }
    }
    stats.avg_degree_l0 = stats.nodes == 0 ? 0.0
                                           : static_cast<double>(stats.edges_level0) /
                                                 static_cast<double>(stats.nodes);
    return stats;
}

}  // namespace ann
