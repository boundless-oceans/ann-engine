#include "ivf_lists.h"

#include <limits>
#include <stdexcept>

#include "ann/types.h"

namespace ann::detail {

IVFLists BuildIVFLists(std::span<const uint32_t> assignments, size_t nlist) {
    if (nlist == 0 || nlist > static_cast<size_t>(kInvalidIdx)) {
        throw std::invalid_argument("IVF requires a positive representable list count");
    }
    if (assignments.size() > static_cast<size_t>(kInvalidIdx)) {
        throw std::length_error("IVF vector count exceeds available IDs");
    }
    if (nlist == std::numeric_limits<size_t>::max()) {
        throw std::length_error("IVF offset array size overflow");
    }
    // 先检查簇 ID，再分配并统计；调用方的分配数组始终不变。
    for (const auto cluster : assignments) {
        if (cluster >= nlist) {
            throw std::invalid_argument("IVF assignment exceeds list count");
        }
    }
    IVFLists lists;
    lists.offsets.resize(nlist + 1, 0);
    lists.ids.resize(assignments.size());
    for (const auto cluster : assignments) {
        ++lists.offsets[static_cast<size_t>(cluster) + 1];
    }
    for (size_t c = 1; c <= nlist; ++c) {
        lists.offsets[c] += lists.offsets[c - 1];
    }
    auto cursor = lists.offsets;
    for (size_t id = 0; id < assignments.size(); ++id) {
        lists.ids[cursor[assignments[id]]++] = static_cast<uint32_t>(id);
    }
    return lists;
}

}  // namespace ann::detail
