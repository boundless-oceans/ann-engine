// 索引抽象基类。所有索引实现（FlatIndex / HNSWIndex / IVFIndex ...）
// 都通过这个接口对外提供服务。
//
// 实现必须满足的契约：
//
//   1. Search 是 const，且不得修改任何成员。这是并发查询的前提。
//      需要临时缓冲时放在栈上，或由调用方传入，不要放成员变量。
//
//   2. Add 与 Search 不得并发调用。约定为单写多读。
//
//   3. MemoryUsage 必须如实统计：向量数据 + 索引结构 + 元数据 + 对齐填充。
//      内存是一等指标，不是事后补的。
#pragma once

// 与包含卫士并用：前者是快路径（编译器直接跳过重复包含），后者在同文件经不同路径被包含时兜底
#ifndef ANN_INDEX_H
#define ANN_INDEX_H

#include <cstddef>
#include <iosfwd>

#include "ann/types.h"

namespace ann {

class Index {
public:
    virtual ~Index() = default;

    // ---------------- 构建 ----------------

    // 追加 n 个向量，每个 dim 维，连续存放（n × dim）。
    virtual void Add(size_t n, const float* vecs) = 0;

    // 需要训练的索引（如 IVF）在此完成训练与结构构建。
    // 无需训练的索引保持默认空实现。
    virtual void Build() {}

    // ---------------- 查询 ----------------

    virtual void Search(const float* query, size_t k, ResultView out) const = 0;

    // 批量查询。默认实现逐条调用 Search；
    // 能利用批量特性的实现（如 SIMD 批处理、GPU）可覆写。
    virtual void SearchBatch(const float* queries, size_t nq, size_t k,
                             ResultView out) const {
        const size_t dim = Dim();
        for (size_t i = 0; i < nq; ++i) {
            ResultView one{out.dists + i * k, out.ids + i * k, k};
            Search(queries + i * dim, k, one);
        }
    }

    // ---------------- 元信息 ----------------

    virtual size_t Size() const = 0;          // 已索引的向量数
    virtual size_t Dim() const = 0;           // 向量维度
    virtual Metric GetMetric() const = 0;     // 距离度量
    virtual size_t MemoryUsage() const = 0;   // 占用字节数
    virtual const char* Name() const = 0;     // 用于报告与 CSV 的索引名

    // ---------------- 持久化（后续阶段）----------------

    // 默认不支持序列化，返回 false。可序列化的索引覆写这两个方法。
    // 给出默认实现而非纯虚函数：避免每个索引都被迫实现尚未用到的能力。
    virtual bool Save(std::ostream& /*os*/) const { return false; }
    virtual bool Load(std::istream& /*is*/) { return false; }
};

}  // namespace ann

#endif  // ANN_INDEX_H
