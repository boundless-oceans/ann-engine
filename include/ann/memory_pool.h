// 对齐分配与连续容器；构建期内存池将在后续单独实现。
#pragma once
#ifndef ANN_MEMORY_POOL_H
#define ANN_MEMORY_POOL_H

#include <algorithm>
#include <limits>
#include <memory>
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
