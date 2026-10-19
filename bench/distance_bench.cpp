// 缓存热数据上的距离微基准：为后续 AVX2 版本提供同条件对照。
#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "ann/distance.h"

namespace {

template<ann::DistFn Distance, bool RequiresAVX2 = false>
void BM_Distance(benchmark::State& state) {
    const auto dim = static_cast<size_t>(state.range(0));
    // 显式标记指令集要求，避免在 GCC 11 上对不同函数地址做常量表达式比较。
    if constexpr (RequiresAVX2) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
            state.SkipWithError("AVX2 and FMA are required");
            return;
        }
#else
        state.SkipWithError("AVX2 implementation requires x86 GCC/Clang");
        return;
#endif
    }
    // 起点按 64 字节对齐，便于后续复用相同数据测 SIMD；不代表当前索引已对齐。
    alignas(64) std::array<float, 960> a{};
    alignas(64) std::array<float, 960> b{};
    if (dim == 0 || dim > a.size()) {
        state.SkipWithError("unsupported dimension");
        return;
    }

    // 初始化在测量循环外。使用固定输入，避免随机数生成与分配污染计时。
    for (size_t i = 0; i < dim; ++i) {
        a[i] = static_cast<float>((i * 17 + 3) % 251) / 251.0f;
        b[i] = static_cast<float>((i * 29 + 11) % 257) / 257.0f;
    }
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    benchmark::DoNotOptimize(a_ptr);
    benchmark::DoNotOptimize(b_ptr);

    for (auto _ : state) {
        // 配合已逃逸的指针，阻止把固定输入的计算整体移出循环或删掉。
        benchmark::ClobberMemory();
        // 使用非 const 左值匹配 Benchmark 的非弃用重载。
        ann::dist_t result = Distance(a_ptr, b_ptr, dim);
        benchmark::DoNotOptimize(result);
    }

    // 一次迭代计算一对向量；字节数是逻辑读取量，不是实际 DRAM 流量。
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(2 * dim * sizeof(float)));
    state.SetLabel("hot-cache; aligned-input; direct-call");
}

BENCHMARK_TEMPLATE(BM_Distance, ann::L2SqrScalar)
    ->Name("Distance/L2/Scalar")
    ->Arg(16)
    ->Arg(127)
    ->Arg(128)
    ->Arg(129)
    ->Arg(960)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_Distance, ann::InnerProductScalar)
    ->Name("Distance/IP/Scalar")
    ->Arg(16)
    ->Arg(127)
    ->Arg(128)
    ->Arg(129)
    ->Arg(960)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_Distance, ann::L2SqrAVX2, true)
    ->Name("Distance/L2/AVX2Single")
    ->Arg(16)
    ->Arg(127)
    ->Arg(128)
    ->Arg(129)
    ->Arg(960)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_Distance, ann::InnerProductAVX2, true)
    ->Name("Distance/IP/AVX2Single")
    ->Arg(16)
    ->Arg(127)
    ->Arg(128)
    ->Arg(129)
    ->Arg(960)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_Distance, ann::L2SqrAVX2Four, true)
    ->Name("Distance/L2/AVX2Four")
    ->Arg(16)
    ->Arg(127)
    ->Arg(128)
    ->Arg(129)
    ->Arg(960)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(BM_Distance, ann::InnerProductAVX2Four, true)
    ->Name("Distance/IP/AVX2Four")
    ->Arg(16)
    ->Arg(127)
    ->Arg(128)
    ->Arg(129)
    ->Arg(960)
    ->Unit(benchmark::kNanosecond);

}  // namespace
