# P1 距离函数微基准

## 当前交付与边界

`ann_distance_bench` 使用 Google Benchmark v1.9.1，直接调用现有的
`L2SqrScalar` 与 `InnerProductScalar`（返回负内积），以及新增的 AVX2 + FMA 单累加器实现。
四累加器已在 `feat/p1-avx2-four-accumulators` 子分支完成 rel/san 测试验证与 128 维三版本同轮测量。
每种函数覆盖 16、127、128、129、960 维，其中 127/129 为后续 SIMD 尾部处理提供对照。
用户已手动完成编译、10 个用例的运行检查及 128 维标量正式测量；基线结果见下文。
AVX2 单累加器已由用户编译、完成 rel/san 各 35 个测试与 128 维正式测量；该阶段注册 20 个微基准。
四累加器交付时总计 41 个测试、注册 30 个微基准。
当前 `feat/p1-distance-dispatch` 新增 9 个派发测试及 10 个间接调用微基准，总计 50 个测试、注册 40 个微基准；rel/san 各 50 个测试通过，128 维同轮微基准已完成，合成与全量 SIFT1M 端到端复测已完成。
后续内存功能已推进至 `feat/p1-build-arena`：当前 rel/san 各 73 个测试通过（沙盒 san 关闭泄漏扫描），距离微基准仍注册 40 项；本块无新性能测量。
以下命令保留用于复现。

微基准反复计算同一对向量，输入起点按 64 字节对齐，数据可以驻留缓存。
初始化、内存准备放在测量循环外；循环内通过 `ClobberMemory` 与 `DoNotOptimize`
防止编译器删除或整体外提计算。它们本身存在固定开销，小维度下尤其需要注意。

`Scalar` 表示现有源码实现，**不保证机器指令完全没有自动向量化**。
保持当前编译优化设置，不启用全局 `-march=native`、`-ffast-math`，不刻意禁用自动向量化。
后续使用同一构建配置比较 SIMD，必要时检查反汇编解释差异。
历史 Scalar/AVX2Single/AVX2Four 条目使用直接调用；新增 Dispatch 条目在计时外选好函数指针，
循环内仅做间接调用。它不把特性探测和函数选择的初始化成本算进每次距离计算。

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

当前派发子分支预期：共 50 个测试通过，列表列出 40 个微基准，dry run 每项仅执行一次。
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
单累加器交付时尚未接入运行时派发，所以当时未重跑全量 SIFT1M；最新派发验证见第 8 节。

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

单累加器交付时新增 8 个测试、合计 35 个；当前子分支四累加器验证见第 7 节。
然后配置 san 并检查尾部加载与其他未定义行为：

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
该次测量时尚未接入索引运行时派发，四累加器保留单累加器对照，三版本结果见第 7 节。

| 新增追问 | 答案要点 |
|---|---|
| 为什么同时检查 AVX2 与 FMA？ | FMA 是独立特性，本实现使用了 fmadd，单检查 AVX2 不足以保证能执行。 |
| 为什么不加全局 -mavx2？ | 避免编译器在标量或其他路径引入高阶指令，给后续安全派发保留基础。 |
| 八维向量累加后如何得到标量？ | 合并高低两个 128 位半部，再水平求和，最后加尾部贡献。 |
| 为什么单累加器之后还要四累加器？ | SIMD 增加每条指令处理的数据量，多累加器进一步减少依赖链对吞吐的限制。 |

## 7. 四累加器子分支与手动验证

分支 `feat/p1-avx2-four-accumulators` 从 P1 集成分支创建，验证通过后由用户合回。
新增入口 `L2SqrAVX2Four` 和 `InnerProductAVX2Four`；现有单累加器入口保留。
两种版本共享 CPU/内存前提：x86 GCC/Clang 上直接调用要求 AVX2 和 FMA，允许非 32 字节对齐输入，
支持任意维度，dim=0 不读取输入。其他平台回退标量，测试与微基准跳过其 SIMD 用例。
该子分支交付时 `SelectDistFn` 仍返回标量，没有修改索引查询路径；后续派发实现见第 8 节。

四累加器主循环每轮处理 32 维，分别累加偏移 0、8、16、24 的八维块。
循环结束按 `(acc0 + acc1) + (acc2 + acc3)` 合并，继续处理完整的八维块，
最后对最多 7 维执行标量尾循环。这样减少主循环中每条 FMA 链的串行依赖，
代价是更多寄存器、合并指令和代码体积；短向量不保证更快，也不承诺相对单累加器达到固定倍数。

新增 6 个测试，检查四块手算、零维/相同向量、32 维及 8 维边界、非对齐、正负混合和内积相消；
一般数值测试复用原 double 参考与误差尺度，精确长度分配配合 ASan 检查尾部过读。
一般输入容差仍为 `2e-5 * max(1, scale)`；L2 的 scale 为参考距离，内积的 scale 为绝对乘积和。
不同版本的累加顺序不同，结果不要求逐 bit 相等。
用户已手动编译并完成 rel/san 验证；已核对完整日志，两套均为 41 个测试通过、0 个失败，
san 缓存确认为 Debug + ASan/UBSan，日志无相关告警。128 维性能测量也已完成，见本节末尾。

### rel / san 检查

依赖与 rel 配置沿用第 2 节，不需要重新下载依赖：

```bash
cmake --build --preset rel --target ann_tests ann_distance_bench -j 8
./build-rel/bin/ann_tests --gtest_filter='DistanceAVX2Four.*'
ctest --preset rel
cmake --preset san -DANN_BUILD_DISTANCE_BENCHMARKS=OFF -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build-deps/googletest-1.15.2"
cmake --build --preset san --target ann_tests -j 8
ctest --preset san
```

支持 AVX2/FMA 的开发机上，预期四累加器 6 个测试通过，rel/san 均总计 41 个通过且无告警。
任一步失败先停止，不继续测性能。

### 同轮三版本对比

先检查 30 个微基准能运行：

```bash
./build-rel/bin/ann_distance_bench --benchmark_list_tests=true
./build-rel/bin/ann_distance_bench --benchmark_dry_run
```

然后记录环境并正式测量，不覆盖之前单累加器原始结果：

```bash
mkdir -p data/p1-results
date -Iseconds > data/p1-results/avx2four-date.txt
lscpu > data/p1-results/avx2four-cpu.txt
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > data/p1-results/avx2four-governor.txt
cat /sys/devices/system/cpu/intel_pstate/no_turbo > data/p1-results/avx2four-no-turbo.txt
./build-rel/bin/ann_distance_bench --benchmark_filter='^Distance/(L2|IP)/(Scalar|AVX2Single|AVX2Four)/128$' --benchmark_min_warmup_time=0.2 --benchmark_min_time=0.5s --benchmark_repetitions=5 --benchmark_out=data/p1-results/avx2four-128.json --benchmark_out_format=json
```

本轮沿用单线程、不显式绑核，最短预热 0.2 秒、每次最短测量 0.5 秒、重复 5 次。
补充记录接电、电源模式，并保持与对照一致；CPU 型号、核心数及频率状态随原始结果保存。
使用同轮 CPU mean 计算 `Scalar/Four` 与 `Single/Four`，同时看 CV。
若 128 维下收益不明显，再用同样的三版本过滤规则测 960 维，输出新文件；
不要预设“必然加速”，也不能仅凭缓存内微基准判断已达到 DRAM 带宽上限。

| 追问 | 答案要点 |
|---|---|
| 四累加器和四线程有何区别？ | 同一线程用四个向量寄存器累加，让 CPU 有机会并行执行独立指令，不创建线程。 |
| 为什么最终不是四个标量分别求和？ | 先合并向量，再做一次水平归约，避免重复水平求和。 |
| 不足 32 维怎么办？ | 继续处理完整八维块，最后处理不足八维的标量尾部，不能丢掉剩余维度或越界加载。 |
| 为什么可能没有预期收益？ | 短向量中合并和调用开销占比高，也可能受加载吞吐、编译器调度或核心/频率变化影响，需测量和剖析。 |

### 四累加器实测结果

2026-10-19 23:57（Asia/Shanghai），原始文件 `data/p1-results/avx2four-128.json`。

| 环境项 | 本轮记录 |
|---|---|
| CPU / 物理核数 / 逻辑线程数 | Intel Core i7-14650HX / 16 / 24（沿用同机信息，JSON 报告 24 个逻辑 CPU） |
| 系统 / 编译器 / 构建 | Ubuntu 22.04.5 LTS / GCC 11.4.0 / RelWithDebInfo（沿用同一构建环境） |
| governor / Turbo | 本轮未保存独立快照；前次 cpu0 powersave、no_turbo=0，不能据此确认本轮状态 |
| 频率调节 | 本轮 Benchmark 警告 CPU scaling enabled；实际持续频率未记录 |
| 线程数 / 绑核 / 输入 | 1 / 未显式绑核 / 128 维、对齐、热缓存 |
| 最短预热 / 每次最短测量 / 重复 | 0.2 秒 / 0.5 秒 / 5 次 |
| 接电 / 电源模式 / 内存容量 | 未记录 / 未记录 / 未记录 |

| CPU mean（ns） | Scalar | AVX2Single | AVX2Four | Scalar/Four | Single/Four |
|---|---:|---:|---:|---:|---:|
| L2² / 128 | 63.6557 | 6.9838 | 5.1599 | 12.34× | 1.35× |
| 负内积 / 128 | 58.1224 | 5.8313 | 5.2830 | 11.00× | 1.10× |

| CPU CV | Scalar | AVX2Single | AVX2Four |
|---|---:|---:|---:|
| L2² / 128 | 1.4954% | 0.1654% | 0.0907% |
| 负内积 / 128 | 0.1744% | 0.1038% | 0.0225% |

按同轮 CPU 均值，四累加器相对单累加器的耗时减少约 26.12%（L2²）、9.40%（负内积）。
这是实测现象，不能单凭该结果确认依赖链是唯一原因；调用开销、加载吞吐、编译器调度与频率/核心差异
都可能影响两种函数的收益，后续仍需反汇编与 perf 分析。该次测量时尚未将四累加器派发给 FlatIndex。
本轮未保存 `avx2four-governor.txt` 和 `avx2four-no-turbo.txt`，表中据实记录缺项，不用事后采样替代过去状态。
后续可复现比较须在测量前保存完整环境；P1 的派发、内存与 perf 工作尚未完成。

## 8. 运行时 CPU 派发与手动验证

当前分支 `feat/p1-distance-dispatch` 从已合入四累加器的 P1 集成分支创建。
用户已完成编译、rel/san 各 50 个测试与 128 维同轮微基准。已核对两套完整 LastTest.log 均为 50 个通过、0 个失败；san 的 CMakeCache 确认 Debug、ASan/UBSan 均 ON，日志无相关告警。合成与全量 SIFT1M 端到端复测已完成，严格 Recall 与 P0 汇总值一致，原 1.0 目标仍未达成。

### 实现与取舍

- `CpuFeatures::Get()` 用局部 const static 缓存一次 CPU 检测结果，不修改共享特性状态。
- x86 GCC/Clang 下通过 builtin 探测 AVX2/FMA；其他平台/编译器暂报告不可用并回退标量。
- `SelectDistFn` 只有在 AVX2 与 FMA 都可用时返回四累加器；不满足时返回标量。
- FlatIndex 已在构造时存储函数指针，这轮无需修改 Search，查询中不重复检测 CPU。
- `SelectDistFnScalar` 返回标量入口，便于独立参考；不提供修改全局后端的开关。
- 余弦仍是负内积临时回退，没有实现归一化；未知枚举值保留此前负内积回退行为。
- 当前按特性选择，不声称所有维度都取得最优速度；短向量的阈值策略需要额外测量与决定。

选择函数指针便于维护多个实现和安全回退，代价是间接调用及可能失去内联机会。
高阶指令继续由函数级 target 属性隔离，不把 -mavx2/-mfma 加给普通检测或调用路径。
局部 static 初始化具备语言层面的线程安全，但这不改变 Add/Search 不可并发的索引契约。
GCC 检测 builtin 的规则见 [官方文档](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/x86-Built-in-Functions.html)。

内部头 `src/distance_dispatch.h` 暴露纯选择函数给测试，不进入公共 include/ann 接口。
新增 9 个测试覆盖无特性、单项特性、双项特性、实际主机缓存、标量约定、独立 double 参考和 Flat 集成。
模拟特性只检查返回地址，不调用潜在不支持的 SIMD 指令；这些测试能验证选择逻辑，
但不能代替在真实无 AVX2 机器上的执行验证。

### 手动 rel / san 验证

```bash
cmake --build --preset rel --target ann_tests ann_bench ann_distance_bench ann_recall_diag -j 8
./build-rel/bin/ann_tests --gtest_filter='DistanceDispatch.*'
ctest --preset rel
cmake --preset san -DANN_BUILD_DISTANCE_BENCHMARKS=OFF -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build-deps/googletest-1.15.2"
cmake --build --preset san --target ann_tests -j 8
ctest --preset san
```

预期派发 9 个测试通过，rel 和 san 各通过 50 个测试；支持 AVX2/FMA 的开发机不应跳过 SIMD 用例。
若 rel 开关被关掉，按第 2 节重新配置；不要删除已有本地依赖。

### 间接调用微基准

Dispatch 条目在测量外选择一次地址，并让指针对优化器不透明；测量循环与直接调用使用相同输入、屏障。
因此比较的是“已选择指针后的距离调用”与直接调用，不是整个 Flat Search，也不包含 CPUID 初始化成本。

```bash
./build-rel/bin/ann_distance_bench --benchmark_list_tests=true
./build-rel/bin/ann_distance_bench --benchmark_dry_run
mkdir -p data/p1-results
date -Iseconds > data/p1-results/dispatch-date.txt
lscpu > data/p1-results/dispatch-cpu.txt
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > data/p1-results/dispatch-governor.txt
cat /sys/devices/system/cpu/intel_pstate/no_turbo > data/p1-results/dispatch-no-turbo.txt
./build-rel/bin/ann_distance_bench --benchmark_filter='^Distance/(L2|IP)/(Scalar|AVX2Four|Dispatch)/128$' --benchmark_min_warmup_time=0.2 --benchmark_min_time=0.5s --benchmark_repetitions=5 --benchmark_out=data/p1-results/dispatch-128.json --benchmark_out_format=json
```

预计列表为 40 项。正式测量仍是单线程未显式绑核，最短预热 0.2 秒、每次最短测量 0.5 秒、重复 5 次，
另行记录接电、电源模式。观察 Dispatch 标签中的实际后端，用同轮 CPU mean 比较 Dispatch/AVX2Four；
没有实测前不声称开销为固定百分比。在标量机器上应与 Scalar 比较，不把被跳过的 AVX2 行用作基线。

### 派发同轮实测结果

2026-10-20 18:15（Asia/Shanghai），原始文件 `data/p1-results/dispatch-128.json`。
文件包含 6 组各 5 次测量，共 30 个 iteration 行，无 error；两组 Dispatch 标签均确认 `indirect-call; AVX2Four`。

| 环境项 | 本轮记录 |
|---|---|
| CPU / 物理核数 / 逻辑线程数 | Intel Core i7-14650HX / 16 / 24（沿用同机信息，JSON 确认 24 个逻辑 CPU） |
| 系统 / 编译器 / 构建 | Ubuntu 22.04.5 / GCC 11.4 / RelWithDebInfo（沿用开发环境，已核对 rel 缓存的构建类型） |
| governor / Turbo | cpu0 powersave / no_turbo=0，允许 Turbo（本轮独立文本快照） |
| 频率调节 / 实际频率 | JSON 报告 scaling enabled；实际持续频率未记录，mhz_per_cpu 不作为实测持续频率 |
| 线程数 / 绑核 / 输入 | 1 / 未显式绑核 / 128 维、对齐、热缓存 |
| 最短预热 / 每次最短测量 / 重复 | 0.2 秒 / 0.5 秒 / 5 次（本轮执行命令） |
| 接电 / 电源模式 / 内存容量 | 未记录 / 未记录 / 未记录 |

| CPU mean（ns） | Scalar | AVX2Four 直接调用 | Dispatch 间接调用 | Scalar/Dispatch | Dispatch 相对直接调用耗时变化 |
|---|---:|---:|---:|---:|---:|
| L2² / 128 | 51.6159 | 5.1551 | 5.6399 | 9.15× | +9.40%（+0.4848 ns） |
| 负内积 / 128 | 47.3261 | 5.2800 | 5.0176 | 9.43× | −4.97%（−0.2623 ns） |

| CPU CV | Scalar | AVX2Four | Dispatch |
|---|---:|---:|---:|
| L2² / 128 | 0.6707% | 0.0815% | 0.0187% |
| 负内积 / 128 | 0.4804% | 0.0260% | 0.0076% |

本轮各组重复测量的波动较小，但内积间接调用反而更快，说明两组耗时之差不能直接视为
固定的函数指针调用成本；代码布局、指令调度、频率和核心迁移等都可能影响结果，
具体原因需要结合反汇编或受控实验，不能仅凭耗时确定。该结果不证明间接调用本身能加速内积。
Scalar/Dispatch 只使用本轮数据计算，不能将历史 Scalar 数字混入本轮比值，
也不能将热缓存单对距离的倍数当成 Flat 检索 QPS 倍数。

### Flat 端到端检查

先用与 P0 相同参数的合成数据打通自动派发后的评测：

```bash
./build-rel/bin/ann_bench --n 100000 --nq 100 --dim 128 --k 10 --metric l2 --warmup 1 --runs 3 | tee data/p1-results/dispatch-synthetic.txt
```

输出新增 `cpu avx2/fma` 与 `distance fn`；本开发机预计为 yes/yes、AVX2Four。
这仍是自算答案，Recall=1 不能代替独立正确性验证。通过后再验证完整 SIFT1M：

```bash
./build-rel/bin/ann_bench --data data --k 10 --metric l2 --warmup 1 --runs 3 | tee data/p1-results/dispatch-sift1m.txt
```

本轮仍为单线程，预热 1 轮、测量 3 轮。保存环境与原始输出，严格按 ID 计算 Recall 的规则不变。
由于 FMA 和归约顺序不同，不事先保证严格 Recall 与 P0 数字完全相同；若出现差异，
使用已重新构建的 ann_recall_diag 复核，不能为追平指标改变计分规则或官方答案。
P0 测量的频率等条件未完全记录，不直接把新 QPS 除以历史 QPS 当作受控加速比。
2026-10-20 用户已完成以上两项评测，原始输出分别为
`data/p1-results/dispatch-synthetic.txt` 与 `data/p1-results/dispatch-sift1m.txt`，两者均报告 yes/yes、AVX2Four。

| 项目 | 合成数据 | 完整 SIFT1M |
|---|---:|---:|
| base / 查询 / 维度 | 100000 / 100 / 128 | 1000000 / 10000 / 128 |
| 严格 Recall@10 | 1.000000（自身生成答案） | 0.999300（官方答案） |
| QPS | 436.2 | 38.8 |
| mean / P50（ms） | 2.2926 / 2.2793 | 25.7523 / 25.0932 |
| P95 / P99 / max（ms） | 2.4128 / 2.4686 / 2.5187 | 28.4243 / 31.7696 / 43.9901 |
| 索引内存（MiB）/ 构建时间（s） | 48.8 / 0.021 | 488.3 / 0.168 |

测量环境沿用 Intel Core i7-14650HX（16 核/24 线程）、Ubuntu 22.04.5、GCC 11.4、
RelWithDebInfo；单线程未显式绑核，L2²、k=10，预热 1 轮、正式测量 3 轮。
端到端执行前未单独保存 governor/Turbo 快照；此前微基准的 powersave、允许 Turbo
不能当作这两项评测执行时已确认的状态。实际频率、接电、电源模式及物理内存容量未记录。

严格 Recall 的汇总值与 P0 一致，原 1.0 验收仍未达成；相同汇总值不证明每条查询返回 ID 完全相同，
本轮没有重新逐查询诊断，也不将 P0 的 70 条等距诊断说成本轮新增验证。
本次输出尚不能定位检索瓶颈；微基准收益没有按同样倍数反映到端到端，后续通过 perf 分析。

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

## 9. perf 采集与瓶颈分析

计数、采样、混合核心选择及环境记录见 [perf 说明](perf.md)。
当前 perf 已安装，但初始 task-clock 探针被拒绝；用户处理后普通终端硬件探针成功，代理环境仍受限，固定 P 核的 Flat 首轮 IPC≈1.36，缓存事件比值58.73%、主要 cycles 样本归属 L2SqrAVX2Four；完整环境与局限见 perf 说明；
已有初步运行时证据，不把高缓存未命中比例直接解释为带宽饱和；Scalar/AVX2 的固定 P 核 IPC 对照也已完成，CPU mean、IPC 和环境见 perf 说明。

## 10. HNSW 端到端基线

运行入口、独立 Flat 参考、环境与结果见 [HNSW 评测说明](hnsw.md)。当前已接入多样性剪枝并记录最近 M 同轮对照；全量 SIFT1M 验收待完成。

同图efSearch扫描、CSV导出、完整SIFT1M曲线与验收范围现已记录在 [HNSW评测说明](hnsw.md)。可选绘图脚本为 plot_hnsw_sweep.py，不增加C++构建依赖。

## 11. Python Faiss 对照

安装隔离、逐条查询计时、完整同轮结果和口径限制见 [Faiss基线](faiss.md)。仅作为Python侧对照，不链接进C++，可选测试为 tests/faiss_bench_test.py。
