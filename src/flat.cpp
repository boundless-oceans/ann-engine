// FlatIndex 的实现：暴力检索，精确解。
//
// 三个设计要点：
//
//   1. 向量连续存放（n × dim 的一维数组）
//      第 i 个向量的第 j 维 = data_[i * dim_ + j]
//      连续存放才谈得上 cache 友好，也是 P1 加 AVX2 的前提 ——
//      SIMD 一次加载 8 个 float，必须保证它们在内存里是挨着的。
//
//   2. Search 全程不分配内存
//      结果缓冲区由调用方提供，我们直接把它当作 top-k 的工作区：
//      先在 out 里建一个大小为 m 的【最大堆】，再逐个比较，
//      比堆顶小就替换堆顶。复杂度 O(n log m)，而不是全排序的 O(n log n)。
//
//   3. Search 是 const，不修改任何成员
//      因此可以被任意多个线程同时调用 —— 这是并发查询的前提。

#include "ann/flat.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ann {
namespace {

// -----------------------------------------------------------------------------
//  在两个【平行数组】上维护最大堆
//
//  std::make_heap / push_heap 只能操作单个区间，而我们需要 dists 与 ids
//  同步交换，所以这里手写【下滤】，两个数组一起动。
//
//  堆顶（下标 0）保存的是「当前保留的 m 个里距离最大」的那个。
//  它的意义是：任何新来的距离只要小于堆顶，就有资格进入前 m 名。
// -----------------------------------------------------------------------------

// 把位置 i 的元素向下调整（堆的有效大小为 size）
void SiftDown(dist_t* dists, idx_t* ids, size_t size, size_t i) {
    for (;;) {
        const size_t left = 2 * i + 1;
        const size_t right = left + 1;

        size_t largest = i;
        if (left < size && dists[left] > dists[largest]) {
            largest = left;
        }
        if (right < size && dists[right] > dists[largest]) {
            largest = right;
        }
        if (largest == i) {
            break;
        }

        std::swap(dists[i], dists[largest]);
        std::swap(ids[i], ids[largest]);
        i = largest;
    }
}

}  // namespace

FlatIndex::FlatIndex(size_t dim, Metric metric)
    : dim_(dim), metric_(metric), dist_fn_(SelectDistFn(metric)) {}

void FlatIndex::Add(size_t n, const float* vecs) {
    if (n == 0 || vecs == nullptr) {
        return;
    }

    // 先验证 ID 和长度，再扩容；失败时索引内容及大小保持不变。
    if (n > static_cast<size_t>(kInvalidIdx) - n_) {
        throw std::length_error("vector count exceeds available IDs");
    }
    const size_t old_size = data_.Size();
    if (dim_ != 0 && n > (std::numeric_limits<size_t>::max() - old_size) / dim_) {
        throw std::length_error("vector storage size overflow");
    }
    const size_t count = n * dim_;
    // 支持把 Data() 的子区间重新 Add：扩容前记偏移，避免原地址失效。
    const auto begin = reinterpret_cast<uintptr_t>(data_.Data());
    const auto source = reinterpret_cast<uintptr_t>(vecs);
    const bool aliases =
        old_size != 0 && source >= begin && (source - begin) / sizeof(float) < old_size;
    size_t offset = 0;
    if (aliases) {
        offset = (source - begin) / sizeof(float);
        if (count > old_size - offset) {
            throw std::invalid_argument("source range exceeds stored vectors");
        }
    }
    data_.Resize(old_size + count);
    if (count != 0) {
        const float* source_data = aliases ? data_.Data() + offset : vecs;
        std::copy_n(source_data, count, data_.Data() + old_size);
    }

    n_ += n;
}

void FlatIndex::Search(const float* query, size_t k, ResultView out) const {
    if (query == nullptr || out.dists == nullptr || out.ids == nullptr) {
        return;
    }

    // 参数 k 与 out.k 都可能被用来表达「要几条」，取较小值最安全
    const size_t want = std::min(k, out.k);
    if (want == 0) {
        return;
    }

    // 库是空的：全部位置填哨兵，调用方据此判断"没有结果"
    if (n_ == 0 || dim_ == 0) {
        for (size_t i = 0; i < want; ++i) {
            out.dists[i] = std::numeric_limits<dist_t>::max();
            out.ids[i] = kInvalidIdx;
        }
        return;
    }

    // 实际能返回的条数：库里可能比 k 还少
    const size_t m = std::min(want, n_);

    // ---- 第一步：用前 m 个向量在 out 里建一个大小为 m 的最大堆 ----
    for (size_t i = 0; i < m; ++i) {
        out.dists[i] = dist_fn_(query, data_.Data() + i * dim_, dim_);
        out.ids[i] = static_cast<idx_t>(i);
    }
    // 自底向上建堆：从最后一个非叶结点开始逐个下滤，复杂度 O(m)
    for (size_t i = m / 2; i > 0; --i) {
        SiftDown(out.dists, out.ids, m, i - 1);
    }

    // ---- 第二步：扫过剩余向量 ----
    // 堆顶是"已保留的 m 个里最差的"。新距离比它小就顶掉它，然后恢复堆序。
    // 绝大多数向量在第 3 行的比较里就被拒掉了，这是 O(n log m) 里那个 log m 很便宜的原因。
    for (size_t i = m; i < n_; ++i) {
        const dist_t d = dist_fn_(query, data_.Data() + i * dim_, dim_);
        if (d < out.dists[0]) {
            out.dists[0] = d;
            out.ids[0] = static_cast<idx_t>(i);
            SiftDown(out.dists, out.ids, m, 0);
        }
    }

    // ---- 第三步：最大堆 → 升序 ----
    // 反复把堆顶（当前最大值）换到有效区末尾并收缩，结果就是升序排列。
    // 这一步只有 O(m log m)，m = k，通常只有 10~100。
    for (size_t end = m; end > 1; --end) {
        std::swap(out.dists[0], out.dists[end - 1]);
        std::swap(out.ids[0], out.ids[end - 1]);
        SiftDown(out.dists, out.ids, end - 1, 0);
    }

    // ---- 第四步：不足 k 条时用哨兵填满 ----
    // bench/metrics.cpp 的 RecallAtK 里有一句 `if (id == kInvalidIdx) continue;`
    // 两处是配套的：这里填哨兵，那里跳过。
    for (size_t i = m; i < want; ++i) {
        out.dists[i] = std::numeric_limits<dist_t>::max();
        out.ids[i] = kInvalidIdx;
    }
}

size_t FlatIndex::Size() const {
    return n_;
}

size_t FlatIndex::Dim() const {
    return dim_;
}

Metric FlatIndex::GetMetric() const {
    return metric_;
}

size_t FlatIndex::MemoryUsage() const {
    // 用 capacity 而不是 size：内存占用看的是实际分配量，不是逻辑长度。
    // 再加上对象本身；不包含 malloc 元数据和内部碎片。
    return data_.CapacityBytes() + sizeof(*this);
}

}  // namespace ann
