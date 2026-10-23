# =============================================================================
#  ann_core —— 索引引擎核心静态库
# =============================================================================
#
#  ⚠️ 源文件必须显式列出，**禁止 file(GLOB)**
#
#     file(GLOB) 只在 configure 阶段求值一次并把结果缓存下来。
#     之后你新增了源文件，CMake 不会重新扫描 —— 表现为：
#       "我明明写了新文件，可编译既不报错、也没把代码链进去"
#     这一条是本项目的硬规矩，见顶层 CMakeLists.txt 的设计原则第 2 条。
#
#  新增源文件时，除了写文件，**还要回来把它加进下面的列表**。
# =============================================================================

add_library(ann_core
    # ---- 基础层 ----
    distance.cpp        # 标量距离函数与当前派发
    distance_avx2.cpp   # AVX2 + FMA 单/四累加器，函数级指令集隔离
    memory_pool.cpp     # 对齐分配与构建期内存池

    # ---- 索引层 ----
    flat.cpp            # FlatIndex：暴力检索 / ground-truth 生成器
    hnsw_graph.cpp      # HNSW：定长邻居槽位与按节点层数分配的上层池
    hnsw_level.cpp      # HNSW：固定种子的随机层数生成
    hnsw_search.cpp     # HNSW：只读贪心下降与单层 ef 搜索
    hnsw.cpp            # HNSW：最近邻连接基线与完整索引查询

    # ---- 尚未实现的模块：实现时取消注释，并加入上面的源文件列表 ----
    # ivf.cpp           倒排索引（IVF）
    # quantizer.cpp     标量量化 / 乘积量化（SQ / PQ）
    # storage.cpp       索引序列化与加载
    # wal.cpp           预写日志与崩溃恢复
)

# 带命名空间的别名。
# 让使用方写 ann::core 而不是裸名 ann_core，语义更清楚，
# 也为将来把 ann_core 拆成多个库留好余地（别名不变，实现可换）。
add_library(ann::core ALIAS ann_core)

# -----------------------------------------------------------------------------
#  头文件路径
#
#  PUBLIC 表示"我自己要、链接我的人也要"。
#  生成器表达式区分两种场景：
#    BUILD_INTERFACE   —— 在本项目源码树里构建时
#    INSTALL_INTERFACE —— 将来被 install 出去、供外部项目 find_package 时
#  目前还没有 install 规则，但提前写好可以避免以后返工。
# -----------------------------------------------------------------------------
target_include_directories(ann_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# -----------------------------------------------------------------------------
#  公共编译选项
#
#  链接 ann_project_options 即继承：C++20 标准、警告集、Sanitizer 旗标。
#  用 PUBLIC 是因为测试和 bench 也都需要这些选项（它们也链接了本库，
#  或者直接链接 ann_project_options，二者均可）。
# -----------------------------------------------------------------------------
target_link_libraries(ann_core
    PUBLIC
        ann_project_options
)
