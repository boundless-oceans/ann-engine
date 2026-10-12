// 数据集加载与合成数据生成。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ann/types.h"

namespace ann::bench {

// 内存中的一份评测数据集。所有向量按行主序连续存放。
struct Dataset {
    size_t n = 0;      // base 向量数
    size_t dim = 0;    // 向量维度
    std::vector<float> base;  // n × dim

    size_t nq = 0;     // 查询数
    std::vector<float> queries;  // nq × dim

    size_t gt_dim = 0;                // groundtruth 每条记录包含的近邻个数
    std::vector<int32_t> groundtruth;  // nq × gt_dim，存放 base 中的下标
};

// ---- fvecs / ivecs 读取 ----
//
// 文件格式为记录序列，每条记录是 [int32 dim][元素 × dim]：
//   fvecs 元素为 float32，ivecs 元素为 int32，均按小端存放。
//
// 返回 false 的情形：文件无法打开、长度为 0、维度非正，
// 或文件长度不是单条记录长度的整数倍（说明文件被截断或损坏）。
bool ReadFvecs(const std::string& path, std::vector<float>* out,
               size_t* dim, size_t* count);
bool ReadIvecs(const std::string& path, std::vector<int32_t>* out,
               size_t* dim, size_t* count);

// ---- 合成数据 ----
//
// 生成 n 个 dim 维的均匀分布随机向量，以及 nq 条查询。
// groundtruth 留空：合成数据没有官方答案，需要用 FlatIndex 自行生成，
// 见 bench_main.cpp 中的 BuildGroundTruth。
Dataset MakeSynthetic(size_t n, size_t nq, size_t dim, uint64_t seed);

// ---- 从磁盘加载 SIFT1M ----
//
// 读取 dir 下的 sift_base.fvecs / sift_query.fvecs / sift_groundtruth.ivecs。
// 任一步失败则返回空 Dataset（n == 0）。
//
// max_base 非 0 时只保留前 max_base 条向量，用于小规模调试。
// 注意：截断 base 之后官方 groundtruth 不再有效，必须重新生成。
Dataset LoadSIFT(const std::string& dir, size_t max_base = 0);

}  // namespace ann::bench
