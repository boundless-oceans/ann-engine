// 对齐分配、连续容器与构建期内存池。
#pragma once
#ifndef ANN_MEMORY_POOL_H
#define ANN_MEMORY_POOL_H

#include <algorithm>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ann/types.h"

namespace ann {

class AlignedAllocator {
public:
    // 对齐必须是至少 sizeof(void*) 的二次幂；零字节返回 nullptr。
    static void* Allocate(size_t bytes, size_t align = kAlignment);
    static void Deallocate(void* p) noexcept;
};

// 非线程安全的构建期单调池，不接入 Search 热路径。
class BuildArena {
public:
    // initial_bytes 是首块大小提示；实际内存按需申请。
    explicit BuildArena(size_t initial_bytes = 1u << 20);
    ~BuildArena();
    BuildArena(const BuildArena&) = delete;
    BuildArena& operator=(const BuildArena&) = delete;
    BuildArena(BuildArena&&) = delete;
    BuildArena& operator=(BuildArena&&) = delete;

    std::pmr::memory_resource* Resource() noexcept;
    // 必须先销毁使用池内存的对象/容器；Reset 不调用这些对象的析构函数。
    // 所有旧分配失效，保留内存块供下一轮复用，Resource 地址不变。
    void Reset() noexcept;
    // 本轮累计成功申请的用户字节数，不因单独 deallocate 而减少。
    size_t UsedBytes() const noexcept;
    // 所有已缓存块的申请字节数，含单调池块内开销，不含缓存元数据。
    size_t CapacityBytes() const noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

namespace detail {

inline void ValidateAlignment(size_t align) {
    if (align < sizeof(void*) || (align & (align - 1)) != 0) {
        throw std::invalid_argument("alignment must be a power of two and at least sizeof(void*)");
    }
}

// 复用标准容器的对象生命周期及异常安全，只替换底层分配策略。
template<typename T>
class VectorAllocator {
public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    explicit VectorAllocator(size_t align = kAlignment) noexcept : alignment(align) {}

    template<typename U>
    VectorAllocator(const VectorAllocator<U>& other) noexcept : alignment(other.alignment) {}

    T* allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(AlignedAllocator::Allocate(count * sizeof(T), alignment));
    }

    void deallocate(T* p, size_t) noexcept { AlignedAllocator::Deallocate(p); }

    template<typename U>
    bool operator==(const VectorAllocator<U>& other) const noexcept {
        return alignment == other.alignment;
    }

    size_t alignment;
};

}  // namespace detail

template<typename T>
class AlignedVector {
public:
    explicit AlignedVector(size_t align = kAlignment) : data_(MakeAllocator(align)) {}

    // 扩容保留原有元素；新增元素值初始化，缩小不收缩已分配容量。
    void Resize(size_t n) { data_.resize(n); }
    T* Data() noexcept { return data_.data(); }
    const T* Data() const noexcept { return data_.data(); }
    size_t Size() const noexcept { return data_.size(); }
    // 统计申请的元素存储，不含 malloc 元数据及分配器内部碎片。
    size_t CapacityBytes() const noexcept { return data_.capacity() * sizeof(T); }

    // 拷贝语义为独立副本；移动语义由标准容器转移存储及对应分配器。

private:
    static detail::VectorAllocator<T> MakeAllocator(size_t align) {
        detail::ValidateAlignment(align);
        return detail::VectorAllocator<T>(std::max(align, alignof(T)));
    }

    std::vector<T, detail::VectorAllocator<T>> data_;
};

}  // namespace ann

#endif  // ANN_MEMORY_POOL_H
