// 评测指标：Recall@K、延迟分位数、QPS 计时。
#pragma once

// 与包含卫士并用：前者是快路径（编译器直接跳过重复包含），后者在同文件经不同路径被包含时兜底
#ifndef ANN_BENCH_METRICS_H
#define ANN_BENCH_METRICS_H

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "ann/types.h"

namespace ann::bench {

// Recall@K。
//
// 对每条查询，统计返回结果中命中真实前 K 近邻的个数，除以 K，再对所有查询取平均。
// 返回 0.0 ~ 1.0 之间的值。
//
//   result_ids   nq × k，行主序；不足 k 条的位置应填 kInvalidIdx
//   groundtruth  nq × gt_dim，行主序；只比较前 min(k, gt_dim) 列
//
// 注意：本函数假定返回结果中不含重复下标。索引实现应保证这一点。
double RecallAtK(const idx_t* result_ids, size_t nq, size_t k,
                 const int32_t* groundtruth, size_t gt_dim);

// 延迟采样与分位数统计。
//
// 只报平均延迟会掩盖长尾，因此这里统计 P50/P90/P95/P99。
// 检索服务的价值很大程度体现在长尾上。
class LatencyRecorder {
public:
    void Record(double milliseconds);

    struct Stats {
        size_t count = 0;
        double mean = 0.0;
        double min = 0.0;
        double p50 = 0.0;
        double p90 = 0.0;
        double p95 = 0.0;
        double p99 = 0.0;
        double max = 0.0;
    };

    // 会复制并排序样本，适合在测量结束后调用一次
    Stats Compute() const;

    void Reset();
    size_t Count() const { return samples_.size(); }

private:
    std::vector<double> samples_;
};

// QPS 计时。
//
// 使用 steady_clock。system_clock 会被系统时间调整影响，
// 而单调时钟才是测量耗时的正确选择。
//
// 支持多段计时：每对 Start/Stop 的时长会累加，
// 因此先跑多轮、最后统一取 Qps() 得到的是所有轮次的整体吞吐。
class QpsTimer {
public:
    void Start();  // 开始一段计时
    void Stop();   // 结束当前段，把本段时长累加进总时长
    void AddQueries(size_t n);

    double ElapsedSeconds() const;  // 所有已结束段的总时长
    double Qps() const;

private:
    std::chrono::steady_clock::time_point start_{};
    std::chrono::steady_clock::time_point stop_{};
    double total_seconds_ = 0.0;
    size_t queries_ = 0;
};

// 给定已升序排序的样本，返回第 pct 百分位（0 ~ 100）。
// 采用「最近秩」法，样本为空时返回 0。
double Percentile(const std::vector<double>& sorted_values, double pct);

// 一行评测结果，用于打印或落盘 CSV
struct BenchRow {
    std::string index_name;
    std::string metric_name;
    size_t n = 0;
    size_t dim = 0;
    size_t k = 0;
    double recall = 0.0;
    double qps = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    size_t memory_bytes = 0;
    double build_seconds = 0.0;
};

}  // namespace ann::bench

#endif  // ANN_BENCH_METRICS_H
