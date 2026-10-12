#include "dataset.h"

#include <cstdio>
#include <cstring>
#include <random>

namespace ann::bench {
namespace {

// 把整个文件读入内存。返回 false 表示打开失败或读取不完整。
//
// fread 不保证一次读完，必须按返回值累加 —— 大文件上这一点尤其重要。
bool ReadWholeFile(const std::string& path, std::vector<uint8_t>* out) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        return false;
    }

    std::fseek(fp, 0, SEEK_END);
    const long bytes = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (bytes <= 0) {
        std::fclose(fp);
        return false;
    }

    out->resize(static_cast<size_t>(bytes));
    size_t got = 0;
    while (got < out->size()) {
        const size_t r = std::fread(out->data() + got, 1, out->size() - got, fp);
        if (r == 0) {
            break;
        }
        got += r;
    }
    std::fclose(fp);
    return got == out->size();
}

// 按 [int32 dim][元素 × dim] 解析记录序列。
//
// 元素一律通过 memcpy 转换，而不是直接把字节缓冲区重新解释成 T*：
// 后者在缓冲区未按 T 对齐时属于未定义行为。
template <typename T>
bool ParseRecords(const std::vector<uint8_t>& raw, std::vector<T>* out,
                  size_t* dim, size_t* count) {
    static_assert(sizeof(T) == 4, "records are 4 bytes per element");

    if (raw.size() < sizeof(int32_t)) {
        return false;
    }

    int32_t d = 0;
    std::memcpy(&d, raw.data(), sizeof(d));
    if (d <= 0) {
        return false;
    }

    const size_t rec = sizeof(int32_t) + static_cast<size_t>(d) * sizeof(T);
    if (raw.size() % rec != 0) {
        return false;
    }

    *dim = static_cast<size_t>(d);
    *count = raw.size() / rec;

    out->clear();
    out->resize((*count) * (*dim));
    for (size_t i = 0; i < *count; ++i) {
        std::memcpy(out->data() + i * (*dim),
                    raw.data() + i * rec + sizeof(int32_t),
                    (*dim) * sizeof(T));
    }
    return true;
}

}  // namespace

bool ReadFvecs(const std::string& path, std::vector<float>* out,
               size_t* dim, size_t* count) {
    std::vector<uint8_t> raw;
    if (!ReadWholeFile(path, &raw)) {
        return false;
    }
    return ParseRecords<float>(raw, out, dim, count);
}

bool ReadIvecs(const std::string& path, std::vector<int32_t>* out,
               size_t* dim, size_t* count) {
    std::vector<uint8_t> raw;
    if (!ReadWholeFile(path, &raw)) {
        return false;
    }
    return ParseRecords<int32_t>(raw, out, dim, count);
}

Dataset MakeSynthetic(size_t n, size_t nq, size_t dim, uint64_t seed) {
    Dataset ds;
    ds.n = n;
    ds.nq = nq;
    ds.dim = dim;

    ds.base.resize(n * dim);
    ds.queries.resize(nq * dim);

    // 固定种子的均匀分布：保证同一组参数每次生成的完全一致，
    // 使性能数字可复现。
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& v : ds.base) {
        v = dist(rng);
    }
    for (float& v : ds.queries) {
        v = dist(rng);
    }

    // groundtruth 故意留空，见头文件说明。
    return ds;
}

Dataset LoadSIFT(const std::string& dir, size_t max_base) {
    Dataset ds;
    size_t n = 0, dim = 0;
    size_t nq = 0, qdim = 0;
    size_t gt_n = 0, gt_dim = 0;

    if (!ReadFvecs(dir + "/sift_base.fvecs", &ds.base, &dim, &n)) {
        return Dataset{};
    }
    if (!ReadFvecs(dir + "/sift_query.fvecs", &ds.queries, &qdim, &nq)) {
        return Dataset{};
    }
    if (!ReadIvecs(dir + "/sift_groundtruth.ivecs", &ds.groundtruth, &gt_dim, &gt_n)) {
        return Dataset{};
    }

    // 三个文件的维度与条数必须自洽，否则说明文件不配套
    if (qdim != dim || gt_n != nq) {
        return Dataset{};
    }

    // 截断 base：仅用于小规模调试。截断后官方 groundtruth 失效。
    if (max_base != 0 && max_base < n) {
        n = max_base;
        ds.base.resize(n * dim);
        ds.groundtruth.clear();
        ds.groundtruth.shrink_to_fit();
        gt_dim = 0;
    }

    ds.n = n;
    ds.dim = dim;
    ds.nq = nq;
    ds.gt_dim = gt_dim;
    return ds;
}

}  // namespace ann::bench
