# =============================================================================
#  单元测试（GoogleTest）
# =============================================================================
#
#  测试策略
#
#    · **手算用例优先**：先用 2~3 个点、2 维的输入，手算出期望结果再写断言。
#      这类用例出错时你能一眼看出是哪一步错了；而"跑 SIFT1M 看召回率"
#      只能告诉你"不对"，不告诉你"哪里不对"。
#
#    · 同一个可执行文件装所有测试：链接一次、启动快，规模大了再按模块拆
#
#    · 用 gtest_discover_tests 而不是 add_test：见文件末尾说明
# =============================================================================


# -----------------------------------------------------------------------------
#  GoogleTest —— 用 FetchContent 拉取，不要求系统预装
#
#  好处：任何人 clone 下来就能构建，不会因为"你机器上没装 gtest"而失败。
#  代价：首次 configure 需要联网，且会把 GTest 源码放进构建目录编译一次。
# -----------------------------------------------------------------------------
include(FetchContent)

# 这两个 cache 变量必须在 MakeAvailable **之前**设置
set(gtest_force_shared_crt ON  CACHE BOOL "" FORCE)  # MSVC 下避免 CRT 冲突
set(INSTALL_GTEST          OFF CACHE BOOL "" FORCE)  # 不要把 GTest 装进系统

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2      # 固定 tag 保证可复现；**不要用 main**
    GIT_SHALLOW    TRUE         # 只拉最新一次提交，省时间
)
FetchContent_MakeAvailable(googletest)


# -----------------------------------------------------------------------------
#  测试可执行文件
#
#  ⚠️ 同 src/CMakeLists.txt：显式列出源文件，新增测试文件时记得回来加一行
# -----------------------------------------------------------------------------
add_executable(ann_tests
    smoke_test.cpp        # 骨架连通性测试：先证明"构建→测试"这条链路是通的
    distance_test.cpp     # 距离函数：手算用例 + 度量派发 + 排序约定
    distance_avx2_test.cpp # SIMD：尾部、非对齐输入、double 参考与误差容限
    distance_dispatch_test.cpp # CPU 检测、组合回退与 Flat 自动派发集成
    memory_pool_test.cpp  # 对齐、扩容、对象生命周期与 Flat 存储集成
    build_arena_test.cpp  # 构建池统计、对齐、重置复用与 pmr 容器生命周期
    hnsw_graph_test.cpp   # 图存储、小图边关系与扩容后的 ID 稳定性
    hnsw_level_test.cpp   # 层数边界、种子复现与百万样本分布
    hnsw_search_test.cpp  # 只读分层搜索、候选堆、访问标记与工作区复用
    flat_test.cpp         # FlatIndex：手算结果、堆替换、边界与批量查询
    # hnsw_test.cpp       HNSW（含 2 维 10 点图结构验证）
)

# 测试私有纯选择函数，不把模拟 CPU 特性的入口暴露为公共 API。
target_include_directories(ann_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(ann_tests
    PRIVATE
        ann::core
        GTest::gtest_main   # 自带 main()，无需自己写入口
)


# -----------------------------------------------------------------------------
#  注册测试
#
#  用 gtest_discover_tests 而不是 add_test 的原因：
#
#    add_test 把**整个可执行文件**当成一条测试，失败时只能看到
#    "ann_tests 失败"，还得手动重跑并加 --gtest_filter 才知道是哪个用例挂了。
#
#    gtest_discover_tests 在构建后扫描可执行文件里的 TEST()/TEST_F()，
#    把每个用例单独注册成一条 ctest 测试。于是：
#      · ctest 输出直接告诉你哪一条挂了
#      · ctest -j 可以并行跑用例
#      · ctest -R <用例名> 能精确重跑失败的用例
# -----------------------------------------------------------------------------
include(GoogleTest)
gtest_discover_tests(ann_tests)
