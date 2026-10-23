#pragma once
#ifndef ANN_SRC_IVF_LISTS_H
#define ANN_SRC_IVF_LISTS_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ann::detail {

// offsets[c] 到 offsets[c+1] 表示第 c 个列表，ids 存原始插入 ID。
struct IVFLists {
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> ids;
};

// 每行仅属于一个簇；列表内按原始 ID 递增，允许空列表。
IVFLists BuildIVFLists(std::span<const uint32_t> assignments, size_t nlist);

}  // namespace ann::detail
#endif  // ANN_SRC_IVF_LISTS_H
