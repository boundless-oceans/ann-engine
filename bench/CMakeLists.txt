# =============================================================================
#  ann_bench —— 端到端评测程序
# =============================================================================
#
#  为什么这里**不用** Google Benchmark
#
#    Google Benchmark 擅长"函数级微基准"——比如比较标量与 AVX2 距离函数
#    在固定输入下的纳秒级差异。那是后续做距离函数优化时的场景。
#
#    而这里要的是**端到端指标**：
#        Recall@K · P50/P95/P99 延迟 · QPS · 内存占用 · 索引构建时间
#    这些都是"跑一批查询、统计结果"的逻辑，自己用 steady_clock 计时更直接，
#    也少一个第三方依赖。
#
#    结论：端到端程序保持独立；P1 的距离微基准按需启用 Google Benchmark。
# =============================================================================

add_executable(ann_bench
    bench_main.cpp        # CLI 入口：解析参数 → 加载数据 → 建索引 → 查询 → 打印指标
    dataset.cpp           # fvecs / bvecs / ivecs 读取 + 合成数据生成器
    metrics.cpp           # Recall@K · 延迟分位（P50/P95/P99）· QPS 计时
)

# bench 内部的头文件互相包含时用 "dataset.h" 这种写法，
# 因此需要把 bench/ 自己加进搜索路径
target_include_directories(ann_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ann_bench
    PRIVATE
        ann::core
)

# 召回差异诊断：仅用于正确性分析，不输出性能指标。
add_executable(ann_recall_diag
    recall_diag.cpp
    dataset.cpp
)
target_link_libraries(ann_recall_diag PRIVATE ann::core)

# 重建误差和排序诊断；保留原向量仅用于离线分析，不进入索引。
add_executable(ann_pq_diag pq_diag.cpp dataset.cpp)
target_link_libraries(ann_pq_diag PRIVATE ann::core)

if(ANN_BUILD_DISTANCE_BENCHMARKS)
    include(FetchContent)

    # 固定依赖版本；关闭第三方自身测试，避免再拉一份 GoogleTest。
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(ann_distance_bench distance_bench.cpp)
    target_link_libraries(ann_distance_bench PRIVATE ann::core benchmark::benchmark_main)
endif()
