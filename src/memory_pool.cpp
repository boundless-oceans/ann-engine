#include "ann/memory_pool.h"

#include <cstdlib>
#include <new>

namespace ann {

void* AlignedAllocator::Allocate(size_t bytes, size_t align) {
    detail::ValidateAlignment(align);
    if (bytes == 0) {
        return nullptr;
    }
    // POSIX 接口不要求 bytes 是 align 的倍数，释放与 std::free 配对。
    void* result = nullptr;
    if (posix_memalign(&result, align, bytes) != 0) {
        throw std::bad_alloc();
    }
    return result;
}

void AlignedAllocator::Deallocate(void* p) noexcept {
    std::free(p);
}

namespace {

// 单调资源 release 时只归还到缓存，直到整个 BuildArena 析构才释放给系统。
class CachedBlocks final : public std::pmr::memory_resource {
public:
    ~CachedBlocks() override {
        for (const auto& block : blocks_) {
            AlignedAllocator::Deallocate(block.data);
        }
    }

    size_t CapacityBytes() const noexcept { return capacity_; }

private:
    struct Block {
        void* data;
        size_t bytes;
        size_t alignment;
        bool in_use;
    };

    void* do_allocate(size_t bytes, size_t alignment) override {
        // 挑选足够大且满足对齐的最小空闲块，避免无谓消耗大块。
        Block* best = nullptr;
        for (auto& block : blocks_) {
            if (!block.in_use && block.bytes >= bytes && block.alignment >= alignment &&
                (best == nullptr || block.bytes < best->bytes)) {
                best = &block;
            }
        }
        if (best != nullptr) {
            best->in_use = true;
            return best->data;
        }
        if (bytes > std::numeric_limits<size_t>::max() - capacity_) {
            throw std::bad_alloc();
        }
        const size_t actual_align = std::max(alignment, kAlignment);
        void* data = AlignedAllocator::Allocate(bytes, actual_align);
        try {
            blocks_.push_back({data, bytes, actual_align, true});
        } catch (...) {
            AlignedAllocator::Deallocate(data);
            throw;
        }
        capacity_ += bytes;
        return data;
    }

    void do_deallocate(void* p, size_t, size_t) override {
        for (auto& block : blocks_) {
            if (block.data == p) {
                block.in_use = false;
                return;
            }
        }
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

    std::vector<Block> blocks_;
    size_t capacity_ = 0;
};

// 统计用户申请，避免把单调资源的对齐填充和块头误当作有效负载。
class CountedResource final : public std::pmr::memory_resource {
public:
    explicit CountedResource(std::pmr::memory_resource* target) : target_(target) {}
    size_t UsedBytes() const noexcept { return used_; }
    void ResetCount() noexcept { used_ = 0; }

private:
    void* do_allocate(size_t bytes, size_t alignment) override {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("alignment must be a nonzero power of two");
        }
        if (bytes > std::numeric_limits<size_t>::max() - used_) {
            throw std::bad_alloc();
        }
        // 零字节仍分配唯一可回收的位置，但用户有效负载计为零。
        void* result = target_->allocate(std::max(bytes, size_t{1}), alignment);
        used_ += bytes;
        return result;
    }

    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
        target_->deallocate(p, std::max(bytes, size_t{1}), alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

    std::pmr::memory_resource* target_;
    size_t used_ = 0;
};

}  // namespace

struct BuildArena::State {
    explicit State(size_t initial_bytes)
        : arena(std::max(initial_bytes, size_t{1}), &blocks), counted(&arena) {}

    // 声明顺序决定逆序析构：先销毁前端，再归还单调块，最后释放缓存。
    CachedBlocks blocks;
    std::pmr::monotonic_buffer_resource arena;
    CountedResource counted;
};

BuildArena::BuildArena(size_t initial_bytes) : state_(std::make_unique<State>(initial_bytes)) {}

BuildArena::~BuildArena() = default;

std::pmr::memory_resource* BuildArena::Resource() noexcept {
    return &state_->counted;
}

void BuildArena::Reset() noexcept {
    state_->arena.release();
    state_->counted.ResetCount();
}

size_t BuildArena::UsedBytes() const noexcept {
    return state_->counted.UsedBytes();
}

size_t BuildArena::CapacityBytes() const noexcept {
    return state_->blocks.CapacityBytes();
}

}  // namespace ann
