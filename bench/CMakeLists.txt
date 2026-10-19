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
#    结论：当前不引入 Google Benchmark；
#          将来做距离函数的微基准时再按需加入。
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
