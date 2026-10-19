# P1 距离函数微基准

## 当前交付与边界

`ann_distance_bench` 使用 Google Benchmark v1.9.1，直接调用现有的
`L2SqrScalar` 与 `InnerProductScalar`（返回负内积），以及新增的 AVX2 + FMA 单累加器实现。
每种函数覆盖 16、127、128、129、960 维，其中 127/129 为后续 SIMD 尾部处理提供对照。
用户已手动完成编译、10 个用例的运行检查及 128 维标量正式测量；基线结果见下文。
AVX2 单累加器已由用户编译、完成 rel/san 共 35 个测试与 128 维正式测量；当前注册 20 个微基准。
以下命令保留用于复现。

微基准反复计算同一对向量，输入起点按 64 字节对齐，数据可以驻留缓存。
初始化、内存准备放在测量循环外；循环内通过 `ClobberMemory` 与 `DoNotOptimize`
防止编译器删除或整体外提计算。它们本身存在固定开销，小维度下尤其需要注意。

`Scalar` 表示现有源码实现，**不保证机器指令完全没有自动向量化**。
保持当前编译优化设置，不启用全局 `-march=native`、`-ffast-math`，不刻意禁用自动向量化。
后续使用同一构建配置比较 SIMD，必要时检查反汇编解释差异。
本轮使用直接调用，不测量运行时函数指针派发开销；后者留到派发接入时单独测量。

## 1. 准备持久依赖目录

所有命令在仓库根目录执行。GoogleTest 继续使用已有的 `build-deps/googletest-1.15.2`。
考虑到此前 GitHub clone 连接不稳定，先手动下载 Benchmark 固定版本源码：

```bash
mkdir -p build-deps
curl -fL --retry 2 --connect-timeout 15 --max-time 180 https://codeload.github.com/google/benchmark/tar.gz/refs/tags/v1.9.1 -o build-deps/benchmark-v1.9.1.tar.gz
tar -xzf build-deps/benchmark-v1.9.1.tar.gz -C build-deps
```

下载失败时停止，不继续解压或配置；已解压的同版本源码可以直接复用。
`build-deps/` 被现有 `build-*/` 规则忽略，不纳入版本控制。

## 2. 配置、构建与检查

```bash
cmake --preset rel -DANN_BUILD_DISTANCE_BENCHMARKS=ON -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build-deps/googletest-1.15.2" -DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK="$PWD/build-deps/benchmark-1.9.1"
cmake --build --preset rel --target ann_distance_bench ann_tests -j 8
ctest --preset rel
./build-rel/bin/ann_distance_bench --benchmark_list_tests=true
./build-rel/bin/ann_distance_bench --benchmark_dry_run
```

预期：共 35 个测试通过，列表列出 20 个微基准，dry run 每项仅执行一次。
AVX2 测试及微基准在不支持 AVX2/FMA 的 CPU 上跳过，不能把跳过视为已验证 SIMD。
dry run 用于检查运行链路，不用于记录性能。无需重跑 SIFT1M。

微基准开关默认为 OFF，不影响未启用它的构建目录。
若 san 仍缓存旧 `/tmp` 依赖路径，后续配置时也需指定持久的 GoogleTest 源码目录。
Sanitizer 构建只检查正确性，不用于测量性能。

## 3. 保存测量环境

```bash
mkdir -p data/p1-results
date -Iseconds > data/p1-results/date.txt
lscpu > data/p1-results/cpu.txt
lscpu -e=CPU,CORE,MAXMHZ > data/p1-results/cpu-topology.txt
free -h > data/p1-results/memory.txt
uname -a > data/p1-results/system.txt
g++ --version > data/p1-results/compiler.txt
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > data/p1-results/governor.txt
cat /sys/devices/system/cpu/intel_pstate/no_turbo > data/p1-results/no-turbo.txt
```

频率或 Turbo 文件不可用时如实记录，不据此假定状态；Intel 的 `no_turbo=0` 表示允许 Turbo。
实际编译器以 CMake 配置摘要为准。CPU 型号、物理核数、逻辑线程数、内存、governor、
Turbo 状态、电源模式、是否接电、是否绑核和核心编号都应随结果保存。
这台机器为混合核心架构，后续对比应选择相同类型的同一逻辑 CPU，并保持电源策略一致。
不要把 lscpu 的最大频率当作本轮持续运行频率。

## 4. 测量 128 维标量基线

先使用单线程、不显式绑核的方式记录初始数据：每项至少预热 0.2 秒，
每次正式测量至少 0.5 秒，重复 5 次，实际迭代数由 Benchmark 自动校准。
环境不稳定时可以出现较大方差，后续优化对照需使用同样设置，或重新测量统一条件的基线。

```bash
./build-rel/bin/ann_distance_bench --benchmark_filter='^Distance/(L2|IP)/Scalar/128$' --benchmark_min_warmup_time=0.2 --benchmark_min_time=0.5s --benchmark_repetitions=5 --benchmark_out=data/p1-results/scalar-128.json --benchmark_out_format=json
```

终端会显示结果，JSON 文件同时保留环境、逐次结果与汇总。查看：

| 字段 | 含义 |
|---|---|
| Time / CPU | 单次距离计算的墙钟耗时 / CPU 耗时，单位 ns；后续对比统一使用同一列 |
| Iterations | Benchmark 为本次测量自动选择的距离计算次数 |
| items_per_second | 每秒完成的向量对距离计算数，不是检索 QPS |
| bytes_per_second | 按两个输入向量长度计算的逻辑字节吞吐，不是实测 DRAM 带宽 |
| mean / median / stddev / cv | 五次测量的均值、中位数、标准差与变异系数 |

运行全部维度时去掉 `--benchmark_filter`，并将输出文件名改为 `scalar-all.json`。
想显式绑核时先核对 CPU 拓扑，再在运行命令前加 `taskset -c <逻辑CPU编号>`，同时记录编号，
并对所有待比较实现重新测量。不要将绑核后的 SIMD 结果直接与本轮未绑核结果计算加速比。

## 5. 首次标量测量记录

2026-10-19 22:43（Asia/Shanghai），原始结果保存在
`data/p1-results/scalar-128.json`（本地，不入库）。采用上述命令：单线程、未显式绑核，
最短预热 0.2 秒、每次测量最短 0.5 秒、重复 5 次，输入 128 维、对齐、缓存内反复使用。

| 环境项 | 本轮记录 |
|---|---|
| CPU / 物理核数 / 逻辑线程数 | Intel Core i7-14650HX / 16 / 24 |
| 系统 / 编译器 / 构建 | Ubuntu 22.04.5 LTS / GCC 11.4.0 / RelWithDebInfo |
| CPU 频率调节 | cpu0 governor 为 powersave；其余 CPU 未逐一采样，未记录实际持续频率 |
| Turbo | `intel_pstate/no_turbo=0`，允许 Turbo |
| 接电 / 电源模式 / 内存容量 | 未记录 / 未记录 / 未记录 |
| 测量库 | Google Benchmark v1.9.1，JSON 中 library_build_type 为 release |

| CPU 耗时统计 | L2² Scalar / 128 | 负内积 Scalar / 128 |
|---|---:|---:|
| mean（ns） | 57.7895 | 48.7095 |
| median（ns） | 58.0292 | 48.5737 |
| stddev（ns） | 1.0831 | 0.8224 |
| CV（无量纲，以百分比表示） | 1.8742% | 1.6883% |

CV 是标准差除以均值，不是纳秒。Google Benchmark 的 JSON 中 CV 行虽然也带有
`time_unit`，但 `aggregate_unit=percentage` 表示这里应按比例解读并乘 100 展示为百分比。
当前波动可作为首次基线记录，不保证未绑核、动态频率下的运行条件完全一致。
后续 AVX2 对比时应在同一轮重新测量 Scalar，以免跨时段的频率与核心迁移干扰加速比。

提取汇总时区分单位：

```bash
python3 - <<'PY'
import json

with open("data/p1-results/scalar-128.json") as f:
    rows = json.load(f)["benchmarks"]
for row in rows:
    if row.get("run_type") != "aggregate":
        continue
    if row.get("aggregate_unit") == "percentage":
        print(f'{row["name"]}: {100 * row["cpu_time"]:.4f}%')
    else:
        print(f'{row["name"]}: {row["cpu_time"]:.4f} {row["time_unit"]}')
PY
```

## 6. AVX2 单累加器实现与手动验证

核心在 `src/distance_avx2.cpp`。每次加载 8 个 float，L2² 做减法后 FMA，负内积做 FMA，
循环结束合并 8 个 lane，再处理不足 8 维的尾部。`loadu` 接受非 32 字节对齐的 float 输入，
不要求 dim 是 8 的倍数；dim=0 不访问输入。

函数通过 `target("avx2,fma")` 局部启用指令集，不给整个库添加全局 ISA 选项。
x86 GCC/Clang 上调用者必须先确认 AVX2 和 FMA 均可用，当前测试和微基准在测量外检查；
其他编译器/平台提供标量回退，微基准会跳过这些平台的 AVX2 条目。
函数属性的作用见 [GCC 文档](https://gcc.gnu.org/onlinedocs/gcc-12.3.0/gcc/x86-Function-Attributes.html)。
运行时派发尚未接入，`SelectDistFn` 仍返回原标量函数，所以这一轮不需要重跑全量 SIFT1M。

设计取舍：单累加器易于验证，保留为四累加器的对照，但存在循环携带的 FMA 依赖链。
先用 loadu 支持通用输入；后续对齐容器和对齐加载的收益应另行测量。
FMA 舍入次数及水平归约顺序与标量不同，不承诺逐 bit 相等。
独立 double 参考测试对 L2 按结果尺度、对内积按绝对乘积和设置容差，避免相消到零时使用不合理的相对误差。
一般输入的容差为 `2e-5 * max(1, scale)`，同时保留精确手算与边界测试。

先构建、运行 rel 测试：

```bash
cmake --build --preset rel --target ann_tests ann_distance_bench -j 8
./build-rel/bin/ann_tests --gtest_filter='DistanceAVX2.*'
ctest --preset rel
```

预期新增 8 个测试通过，合计 35 个。然后配置 san 并检查尾部加载与其他未定义行为：

```bash
cmake --preset san -DANN_BUILD_DISTANCE_BENCHMARKS=OFF -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build-deps/googletest-1.15.2"
cmake --build --preset san --target ann_tests -j 8
ctest --preset san
```

通过后再检查并正式测量，使用 rel 二进制：

```bash
./build-rel/bin/ann_distance_bench --benchmark_dry_run
mkdir -p data/p1-results
lscpu > data/p1-results/avx2single-cpu.txt
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > data/p1-results/avx2single-governor.txt
cat /sys/devices/system/cpu/intel_pstate/no_turbo > data/p1-results/avx2single-no-turbo.txt
./build-rel/bin/ann_distance_bench --benchmark_filter='^Distance/(L2|IP)/(Scalar|AVX2Single)/128$' --benchmark_min_warmup_time=0.2 --benchmark_min_time=0.5s --benchmark_repetitions=5 --benchmark_out=data/p1-results/avx2single-128.json --benchmark_out_format=json
```

本轮仍为单线程、不显式绑核，最短预热 0.2 秒、每次最短测量 0.5 秒、重复 5 次。
接电、电源模式、CPU 型号/核心数及频率策略随结果记录。
加速比使用本轮 Scalar mean / AVX2Single mean（CPU 列），不要直接拿上次标量数字作为分子。
比较其他维度时去掉 filter 并使用新的输出文件名；本轮 128 维结果记录如下。

### 单累加器实测结果

2026-10-19 23:42（Asia/Shanghai），用户已完成上述 128 维对比，原始 JSON 为
`data/p1-results/avx2single-128.json`。rel、san 的完整测试日志均为 35 个通过、0 个失败，
未发现 Sanitizer 告警。编译时遇到 GCC 11 函数地址常量比较错误，已改为显式布尔模板标记；
同时将 `DoNotOptimize` 的结果参数改为非 const 左值，消除弃用警告。
本轮 Scalar 与 AVX2 使用修正后的同一套测量框架，历史标量记录仅供参考。

| 环境项 | 本轮记录 |
|---|---|
| CPU / 物理核数 / 逻辑线程数 | Intel Core i7-14650HX / 16 / 24 |
| 系统 / 编译器 / 构建 | Ubuntu 22.04.5 LTS / GCC 11.4.0 / RelWithDebInfo |
| governor / Turbo | cpu0 powersave，其余 CPU 未逐一采样 / no_turbo=0，允许 Turbo |
| 查询线程 / 绑核 / 输入 | 1 / 未显式绑核 / 128 维、对齐、热缓存 |
| 最短预热 / 每次最短测量 / 重复 | 0.2 秒 / 0.5 秒 / 5 次 |
| 接电 / 电源模式 / 内存容量 / 实际持续频率 | 未记录 / 未记录 / 未记录 / 未记录 |

| 函数 | Scalar mean（CPU ns） | AVX2Single mean（CPU ns） | 同轮加速比 | Scalar CV / AVX2 CV |
|---|---:|---:|---:|---:|
| L2² / 128 | 63.8192 | 6.8118 | 9.37× | 1.6757% / 0.9371% |
| 负内积 / 128 | 62.2633 | 5.9057 | 10.54× | 1.0827% / 0.6697% |

这些倍数只说明该机器、当前编译与动态频率条件下，缓存内直接调用的实测收益。
未绑核和允许 Turbo 仍可能影响不同函数之间的运行条件；CV 较小不等于排除了系统偏差。
超过 8× 不代表硬件向量宽度变大或已达到计算/带宽上限，完整归因需要后续反汇编及 perf 分析。
输出的约 140–161 GiB/s 是逻辑读取吞吐，不是 DRAM 实测带宽，也不能将加速比直接套到 Flat QPS。
目前尚未接入索引运行时派发，下一步保留单累加器对照并实现四累加器。

| 新增追问 | 答案要点 |
|---|---|
| 为什么同时检查 AVX2 与 FMA？ | FMA 是独立特性，本实现使用了 fmadd，单检查 AVX2 不足以保证能执行。 |
| 为什么不加全局 -mavx2？ | 避免编译器在标量或其他路径引入高阶指令，给后续安全派发保留基础。 |
| 八维向量累加后如何得到标量？ | 合并高低两个 128 位半部，再水平求和，最后加尾部贡献。 |
| 为什么单累加器之后还要四累加器？ | SIMD 增加每条指令处理的数据量，多累加器进一步减少依赖链对吞吐的限制。 |

## 关键取舍与面试讲点

本轮先测缓存内计算，能更直接比较算术与依赖链；代价是无法代表百万向量扫描，
后续仍要结合 Flat 端到端评测与 perf 判断访存瓶颈。
选择固定版本 Google Benchmark 以复用自动校准、预热、重复测量和结构化输出；
代价是新增一项可选构建依赖。

| 追问 | 答案要点 |
|---|---|
| 为什么不只比较 Flat QPS？ | Flat 还包含扫描、top-k 与内存访问，微基准先隔离单对向量计算成本。 |
| 为什么输入初始化放循环外？ | 避免将分配和数据生成算作距离函数耗时。 |
| 为什么防止编译器优化掉结果？ | 未使用的结果或固定输入可能被删除、外提，导致测到错误的低耗时。 |
| 为什么同时测 127、128、129 维？ | 128 覆盖 SIFT，邻近维度用于观察 SIMD 尾部处理带来的成本。 |
| B/s 很高是否说明突破内存带宽？ | 这里输入可驻留缓存，计数是逻辑读取量，不是 DRAM 实测流量。 |
