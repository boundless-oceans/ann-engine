# =============================================================================
#  ann-engine —— 顶层构建脚本
# =============================================================================
#
#  设计原则（三条，后面所有 CMakeLists 都遵守）
#
#    1. 全部使用 target-based 写法（target_xxx）
#       不用全局的 include_directories() / add_definitions() / link_libraries()
#       —— 全局命令会污染其后定义的**所有** target，且看不出依赖从哪来
#
#    2. 源文件显式列出，禁止 file(GLOB)
#       GLOB 只在 configure 时求值一次，新增源文件不会被感知，
#       表现为"我明明写了新文件，编译既不报错也没链接进去"
#
#    3. 每层一个 CMakeLists.txt，只负责定义"本层产生的 target"
#       顶层管全局设置与 add_subdirectory，子目录管自己的库/可执行文件
#
#  ---------------------------------------------------------------------------
#  常用命令（等价于 CMakePresets.json 里的预设）
#
#    cmake --preset rel                 # 配置：RelWithDebInfo
#    cmake --build --preset rel         # 构建，预设最多 8 个并行任务
#    ctest --preset rel                 # 跑测试
#
#    # 不依赖 presets 的等价手写形式：
#    cmake -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
#    cmake --build build-rel -j 8
#    ctest --test-dir build-rel --output-on-failure
#  ---------------------------------------------------------------------------
# =============================================================================


# -----------------------------------------------------------------------------
#  CMake 版本
#
#  3.22 = Ubuntu 22.04 自带的版本，也就是本项目主力开发环境的版本下限。
#
#  写成 "最低...最高" 区间：
#    · 老 CMake（3.22）按 3.22 的策略执行
#    · 新 CMake（如 4.x）按 3.28 的策略执行
#  好处是既能在旧环境构建，又能让新 CMake 启用更严格的策略、提前暴露问题。
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.22...3.28)


# -----------------------------------------------------------------------------
#  项目声明
# -----------------------------------------------------------------------------
project(ann-engine
    VERSION      0.1.0
    DESCRIPTION  "From-scratch ANN (approximate nearest neighbor) search engine"
    HOMEPAGE_URL "https://github.com/boundless-oceans/ann-engine"
    LANGUAGES    CXX
)


# -----------------------------------------------------------------------------
#  全局开关
#
#  约定：本项目自己的所有选项统一加 ANN_ 前缀。
#  不加前缀的选项（BUILD_TESTING 等）是 CMake 或第三方定义的标准选项，
#  自定义选项加前缀可以避免和它们撞名。
# -----------------------------------------------------------------------------
option(ANN_WARNINGS_AS_ERRORS "把编译警告视为错误（建议只在 CI 中开启）" OFF)


# -----------------------------------------------------------------------------
#  标准测试开关
#
#  用 include(CTest) 提供的标准选项 BUILD_TESTING，不自造 ANN_BUILD_TESTS ——
#  遵循标准能少一次解释成本，CI 和 IDE 也都认得这个开关。
#  include(CTest) 同时会调用 enable_testing()，使 ctest 可用。
# -----------------------------------------------------------------------------
include(CTest)


# -----------------------------------------------------------------------------
#  全局构建设置
# -----------------------------------------------------------------------------

# 导出 compile_commands.json —— clang-tidy / clangd / IDE 都依赖它
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 产物统一放到 <build>/bin 与 <build>/lib
# 好处：文档、脚本、.gitignore 里引用路径时不用猜
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# CMake 4.0 起移除了对 cmake_minimum_required(VERSION < 3.5) 的兼容，
# 而不少第三方项目（含部分版本的 GoogleTest 依赖）仍写着更低的版本要求。
# 这个变量是官方给出的兜底开关，只在新版 CMake 上设置。
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING
        "兼容仍声明 CMake < 3.5 的第三方项目" FORCE)
endif()


# -----------------------------------------------------------------------------
#  ccache —— 改一个文件不用重编全部
#
#  必须在创建 target **之前**设置，因为 CMake 在生成构建规则时会读取它。
#  注意：ccache 只在重复构建时有效；CI 里要配缓存才有意义。
# -----------------------------------------------------------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()


# -----------------------------------------------------------------------------
#  ann_project_options —— 所有 target 共用的编译选项
#
#  用 INTERFACE 库承载公共编译选项，是本项目 target-based 写法的核心：
#  每个 target 只需 target_link_libraries(... ann_project_options) 一次，
#  不必重复写几十行 target_compile_options。
#
#  INTERFACE 库不产生任何编译产物，只是"一组要往下传递的用法要求"。
# -----------------------------------------------------------------------------
add_library(ann_project_options INTERFACE)

# C++20 —— 通过 compile feature 表达，而不是硬写 -std=c++20：
# 这样换编译器时 CMake 会自动翻译成对应的旗标（MSVC 是 /std:c++20）
target_compile_features(ann_project_options INTERFACE cxx_std_20)

# ---- 警告：按编译器分派，保证同一份工程在 GCC / Clang / MSVC 下都能构建 ----
if(MSVC)
    target_compile_options(ann_project_options INTERFACE
        /W4             # 微软建议的高警告级别
        /permissive-    # 关闭宽松模式，贴近标准
        /Zc:__cplusplus # 让 __cplusplus 反映真实的 C++ 标准（否则恒为 199711L）
        /utf-8          # 源码与执行字符集都用 UTF-8
                        # （否则 MSVC 按系统代码页读源文件，中文注释与字符串会乱）
    )
else()
    target_compile_options(ann_project_options INTERFACE
        -Wall
        -Wextra
        -Wpedantic

        # 与 C++ 语义相关的
        -Wshadow                  # 内层作用域遮蔽外层同名变量
        -Wnon-virtual-dtor        # 基类有虚函数但析构函数非虚
        -Woverloaded-virtual      # 派生类"覆写"实为隐藏了基类的虚函数
        -Wold-style-cast          # C 风格强制转换（本项目大量用 static_cast）
        -Wcast-align              # 转换后对齐要求变严（SIMD / 指针运算里常见）

        # 数值精度 —— 对 SIMD 代码尤为重要
        -Wdouble-promotion        # float 被隐式提升成 double（会拖慢向量化）
        -Wfloat-equal             # 用 == 比较浮点数

        # 其他
        -Wnull-dereference
        -Wformat=2                # printf 系列格式串检查（bench 打印要常用）
        -Wunused
    )
endif()

# 是否把警告当错误（默认关：开发期频繁改代码时太碍事；CI 里建议开）
if(ANN_WARNINGS_AS_ERRORS)
    if(MSVC)
        target_compile_options(ann_project_options INTERFACE /WX)
    else()
        target_compile_options(ann_project_options INTERFACE -Werror)
    endif()
endif()


# -----------------------------------------------------------------------------
#  Sanitizer（动态检查）
#
#  ⚠️ 三条铁律
#     1. ASan 与 TSan 互斥 —— 两者都要接管内存分配器，同时开必然冲突
#     2. Sanitizer 构建**只能**用于正确性检查，绝不可用来测性能
#        （开销是数倍，测出来的 QPS 毫无意义）
#     3. 因此 Sanitizer 用独立的构建目录（build-san / build-tsan），与 build-rel 隔离
# -----------------------------------------------------------------------------
option(ANN_ENABLE_ASAN  "启用 AddressSanitizer（越界 / use-after-free / 泄漏）" OFF)
option(ANN_ENABLE_TSAN  "启用 ThreadSanitizer（数据竞争）"                      OFF)
option(ANN_ENABLE_UBSAN "启用 UndefinedBehaviorSanitizer（未定义行为）"          OFF)

if(ANN_ENABLE_ASAN AND ANN_ENABLE_TSAN)
    message(FATAL_ERROR
        "ASan 与 TSan 不能同时启用：两者都要接管内存分配器，会互相冲突。\n"
        "请用独立目录分别构建：build-san 与 build-tsan。")
endif()

set(ANN_SANITIZER_LIST "")
if(ANN_ENABLE_ASAN)
    list(APPEND ANN_SANITIZER_LIST address)
endif()
if(ANN_ENABLE_TSAN)
    list(APPEND ANN_SANITIZER_LIST thread)
endif()
if(ANN_ENABLE_UBSAN)
    list(APPEND ANN_SANITIZER_LIST undefined)
endif()

if(ANN_SANITIZER_LIST)
    if(MSVC)
        # MSVC 只有 /fsanitize=address，且不支持 TSan / UBSan
        message(WARNING "MSVC 仅支持 ASan；当前配置未在此工具链上适配")
    else()
        # 把列表拼成 -fsanitize=a,b,c 的形式（GCC 与 Clang 都接受逗号分隔）
        string(REPLACE ";" "," ANN_SANITIZE_ARG "${ANN_SANITIZER_LIST}")

        target_compile_options(ann_project_options INTERFACE
            -fsanitize=${ANN_SANITIZE_ARG}
            -fno-omit-frame-pointer   # 保留帧指针，否则栈回溯不可读
            -fno-sanitize-recover=all # 出现问题直接终止，便于 CI 判定失败
            -g
        )
        # Sanitizer 的运行时库也要链接进去，否则运行时符号缺失
        target_link_options(ann_project_options INTERFACE
            -fsanitize=${ANN_SANITIZE_ARG}
        )
    endif()
endif()


# -----------------------------------------------------------------------------
#  子目录
#
#  顺序即依赖顺序：库在前，用它的可执行文件在后
# -----------------------------------------------------------------------------
add_subdirectory(src)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(bench)


# -----------------------------------------------------------------------------
#  配置摘要
#
#  配置完成后回显关键设置。有两个实际用途：
#    1. 发现"我以为开了 ASan 其实没开"这类静默错误
#    2. 性能数据必须连同环境一起记录，
#       这份摘要就是"编译环境"那一行
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==================== ann-engine configuration ====================")
message(STATUS "  version         : ${PROJECT_VERSION}")
message(STATUS "  build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ standard    : 20")
message(STATUS "  build directory : ${CMAKE_BINARY_DIR}")
message(STATUS "  build tests     : ${BUILD_TESTING}")
message(STATUS "  warnings-as-err : ${ANN_WARNINGS_AS_ERRORS}")
message(STATUS "  sanitizers      : ${ANN_SANITIZER_LIST}")
message(STATUS "  ccache          : ${CCACHE_PROGRAM}")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "  WARNING: CMAKE_BUILD_TYPE is empty. With a single-config generator")
    message(STATUS "           this produces an unoptimized build; timings are invalid.")
endif()
message(STATUS "==================================================================")
message(STATUS "")
message(STATUS "")
