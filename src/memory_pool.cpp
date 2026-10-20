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

}  // namespace ann
