#include "metrics.h"

#include <algorithm>
#include <cmath>

namespace ann::bench {

double RecallAtK(const idx_t* result_ids, size_t nq, size_t k,
                 const int32_t* groundtruth, size_t gt_dim) {
    if (result_ids == nullptr || groundtruth == nullptr || nq == 0 || gt_dim == 0 || k == 0) {
        return 0.0;
    }

    // 两侧长度不等时只比较公共部分
    const size_t kk = std::min(k, gt_dim);
    if (kk == 0) {
        return 0.0;
    }

    double hit_total = 0.0;
    for (size_t q = 0; q < nq; ++q) {
        const idx_t* got = result_ids + q * k;
        const int32_t* gt = groundtruth + q * gt_dim;

        size_t hit = 0;
        for (size_t i = 0; i < kk; ++i) {
            const idx_t id = got[i];
            if (id == kInvalidIdx) {
                continue;  // 索引返回条数不足
            }
            for (size_t j = 0; j < kk; ++j) {
                if (static_cast<int64_t>(gt[j]) == static_cast<int64_t>(id)) {
                    ++hit;
                    break;
                }
            }
        }
        hit_total += static_cast<double>(hit) / static_cast<double>(kk);
    }

    return hit_total / static_cast<double>(nq);
}

void LatencyRecorder::Record(double milliseconds) {
    samples_.push_back(milliseconds);
}

void LatencyRecorder::Reset() {
    samples_.clear();
}

LatencyRecorder::Stats LatencyRecorder::Compute() const {
    Stats s;
    s.count = samples_.size();
    if (samples_.empty()) {
        return s;
    }

    std::vector<double> v = samples_;
    std::sort(v.begin(), v.end());

    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }

    s.mean = sum / static_cast<double>(v.size());
    s.min = v.front();
    s.max = v.back();
    s.p50 = Percentile(v, 50.0);
    s.p90 = Percentile(v, 90.0);
    s.p95 = Percentile(v, 95.0);
    s.p99 = Percentile(v, 99.0);
    return s;
}

double Percentile(const std::vector<double>& sorted_values, double pct) {
    if (sorted_values.empty()) {
        return 0.0;
    }
    if (pct <= 0.0) {
        return sorted_values.front();
    }
    if (pct >= 100.0) {
        return sorted_values.back();
    }

    const double n = static_cast<double>(sorted_values.size());
    // 最近秩法：rank = ceil(pct / 100 * n)，下标从 1 开始，故减 1
    double rank = std::ceil(pct / 100.0 * n);
    if (rank < 1.0) {
        rank = 1.0;
    }
    const size_t idx = static_cast<size_t>(rank) - 1;
    return sorted_values[std::min(idx, sorted_values.size() - 1)];
}

void QpsTimer::Start() {
    start_ = std::chrono::steady_clock::now();
}

void QpsTimer::Stop() {
    stop_ = std::chrono::steady_clock::now();
    // 累加而不是覆盖：调用方可能分多轮测量，最后一次性取吞吐
    total_seconds_ += std::chrono::duration<double>(stop_ - start_).count();
}

void QpsTimer::AddQueries(size_t n) {
    queries_ += n;
}

double QpsTimer::ElapsedSeconds() const {
    return total_seconds_;
}

double QpsTimer::Qps() const {
    const double sec = ElapsedSeconds();
    if (sec <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(queries_) / sec;
}

}  // namespace ann::bench
