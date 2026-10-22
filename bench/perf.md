# P1 perf 采集与解读

## 当前状态

2026-10-22：perf 6.8.12 已安装，内核 6.8.0-138-generic。
初始代理沙盒和用户普通终端的 task-clock 探针均被拒绝，paranoid=4。用户临时调整到 2 后，普通终端固定 CPU 0 的项目 IPC 采集已成功。代理复查仍受限，后续采集由用户普通终端完成。
没有修改 sysctl、文件 capabilities 或执行提权；权限机制见 [Linux 内核文档](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html)。

当前 CPU 为 Intel Core i7-14650HX（16 个物理核，24 个逻辑 CPU），
sysfs 报告 cpu_core=0-15、cpu_atom=16-23，代理允许 CPU=0-23。
因此下面选 CPU 0，固定在同一种核心上；机器/允许集合变化后须重新确认。
当前读到 cpu0 governor=powersave、no_turbo=0；这只是准备阶段快照，不能替代实际采集前快照。

BuildArena 用户已报告普通终端默认 san 检查通过，73 个测试全部通过；
代理此前沙盒运行仍是 detect_leaks=0，不混淆两次执行条件。
P1 perf 验收尚未完成，不因代码已完成或微基准达标而标记整个 P1 完成。

## 先检查采集权限

在普通终端执行：

```bash
perf stat -e task-clock -- true
```

普通终端初始被拒绝、处理后硬件探针已成功；权限步骤参见 [学习教程的权限章节](../docs/learn/perf-analysis.md#第一步解决当前权限)，结束后可恢复；代理不修改系统设置。
若仅代理沙盒被拒绝，由用户在普通终端采集，代理读取产物分析。
本文件不自动修改系统级安全策略。

## 保存本轮环境

在仓库根目录执行，使用新结果目录，避免覆盖旧测量。
以下 shell 变量需在同一个终端中保留。

```bash
mkdir -p data/p1-results
perf_result_dir=$(mktemp -d "$PWD/data/p1-results/perf-XXXXXX")
date -Iseconds > "$perf_result_dir/date.txt"
git rev-parse HEAD > "$perf_result_dir/commit.txt"
git status --short > "$perf_result_dir/worktree.txt"
lscpu > "$perf_result_dir/cpu.txt"
uname -r > "$perf_result_dir/kernel.txt"
perf --version > "$perf_result_dir/perf-version.txt"
cat /sys/bus/event_source/devices/cpu_core/cpus > "$perf_result_dir/core-cpus.txt"
cat /sys/bus/event_source/devices/cpu_atom/cpus > "$perf_result_dir/atom-cpus.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$perf_result_dir/governor.txt"
cat /sys/devices/system/cpu/intel_pstate/no_turbo > "$perf_result_dir/no-turbo.txt"
cat /proc/sys/kernel/perf_event_paranoid > "$perf_result_dir/perf-event-paranoid.txt"
```

另行记录接电、电源模式及后台负载。使用 GCC 11.4 RelWithDebInfo，禁止用 san 进行性能测量。
如修改过源码，先 `CCACHE_DISABLE=1 cmake --build --preset rel -j 8`。
这里不修改 governor/Turbo，不把 Benchmark 显示的 MHz 当成持续频率实测。

## 1. 距离内核的 IPC

单线程，CPU 0，128 维、对齐热缓存输入；最短预热 0.2 秒、测量 0.5 秒，重复 5 次。
分别跑 Scalar 与 AVX2Four；先只对 L2 做分析，避免混合不同内核的事件计数。

```bash
taskset -c 0 perf stat -e '{cycles:u,instructions:u}' \
  -o "$perf_result_dir/l2-scalar-stat.txt" -- \
  ./build-rel/bin/ann_distance_bench \
  --benchmark_filter='^Distance/L2/Scalar/128$' \
  --benchmark_min_warmup_time=0.2 --benchmark_min_time=0.5s \
  --benchmark_repetitions=5 \
  --benchmark_out="$perf_result_dir/l2-scalar.json" --benchmark_out_format=json \
  > "$perf_result_dir/l2-scalar-console.txt" 2>&1

taskset -c 0 perf stat -e '{cycles:u,instructions:u}' \
  -o "$perf_result_dir/l2-avx2four-stat.txt" -- \
  ./build-rel/bin/ann_distance_bench \
  --benchmark_filter='^Distance/L2/AVX2Four/128$' \
  --benchmark_min_warmup_time=0.2 --benchmark_min_time=0.5s \
  --benchmark_repetitions=5 \
  --benchmark_out="$perf_result_dir/l2-avx2four.json" --benchmark_out_format=json \
  > "$perf_result_dir/l2-avx2four-console.txt" 2>&1
```

计数器覆盖整个进程，包括初始化、校准、warmup 和输出，并非仅被计时循环。
Scalar 与 AVX2 自动校准的迭代数不同，不能直接比较两份总 instructions 认定单次计算减少多少指令。
IPC 可作为整体执行特征观察，也不能单凭 IPC 判断哪种实现耗时更短。
若混合 PMU 输出多个事件行，只解读实际运行的 cpu_core，不能将不适用的 atom 行当作零。

## 2. Flat 的计数与热点

单线程、CPU 0，合成 100000×128，1000 条查询，k=10，warmup=1、runs=3。
synthetic 另有生成自身答案和召回复算，因此整进程计数包含额外检索以及分配/建索引。
这里用于定位整体热点，不把采集运行的 QPS 当作无插桩的正式性能基线。

```bash
taskset -c 0 perf stat -e '{cycles:u,instructions:u}' \
  -o "$perf_result_dir/flat-ipc.txt" -- \
  ./build-rel/bin/ann_bench --n 100000 --nq 1000 --dim 128 \
  --k 10 --metric l2 --warmup 1 --runs 3 \
  > "$perf_result_dir/flat-ipc-console.txt" 2>&1

taskset -c 0 perf stat -e '{cache-references:u,cache-misses:u}' \
  -o "$perf_result_dir/flat-cache.txt" -- \
  ./build-rel/bin/ann_bench --n 100000 --nq 1000 --dim 128 \
  --k 10 --metric l2 --warmup 1 --runs 3 \
  > "$perf_result_dir/flat-cache-console.txt" 2>&1

taskset -c 0 perf record -e cycles:u -F 199 --call-graph dwarf,8192 \
  -o "$perf_result_dir/flat.perf.data" -- \
  ./build-rel/bin/ann_bench --n 100000 --nq 1000 --dim 128 \
  --k 10 --metric l2 --warmup 1 --runs 3 \
  > "$perf_result_dir/flat-record-console.txt" 2>&1

perf report --stdio --no-children --percent-limit 1 \
  -i "$perf_result_dir/flat.perf.data" > "$perf_result_dir/flat-report.txt"
printf '%s\n' "$perf_result_dir"
```

拆成两组 stat，减少计数器复用；检查 not supported、not counted 和实际运行时间比例。
若 group 无法调度或事件不支持，应记录错误并调整采集，不能把缺失事件当作零。
DWARF 采样无需修改现有编译参数，代价是采样开销和文件体积；这里先用较低采样频率。

## 如何得出结论

| 观察 | 可以说什么 | 还不能说什么 |
|---|---|---|
| instructions/cycles | 当前采集范围的平均 IPC | IPC 高一定更快，或已达到理论算力上限 |
| cache-misses/cache-references | 当前 PMU 通用缓存事件的未命中比例 | L1/L2/LLC 各层精确未命中率或 DRAM 带宽已饱和 |
| 距离函数占较多 cycles 样本 | 样本集中在距离计算代码 | 唯一原因是算术依赖，访存停顿也可能发生在该函数 |
| Flat/SiftDown 样本明显 | 扫描/top-k 值得进一步拆解 | 不经测量就替换数据结构 |

P1 的目标是用实测支持瓶颈判断，不提前写“已经撞到内存带宽墙”。
先解释这组较短工作负载，再决定是否需要全量 SIFT1M 的采样或专用内存事件。
当前IPC、缓存事件与 cycles 热点首轮采集均已完成，初步解读见末节。


## 首轮 Flat IPC 结果

2026-10-22 12:46:08（Asia/Shanghai），结果目录 `data/p1-results/perf-OGEgW0`。
原始统计 `flat-ipc.txt`，程序输出 `flat-console.txt`。同目录 date.txt 为 12:44:26 的准备时间，
第一次权限失败后在 12:46:08 成功重采，不把 date.txt 当成正式统计开始时间。

| 项目 | 本轮记录 |
|---|---|
| CPU / 核心数 | Intel Core i7-14650HX / 16 物理核、24 逻辑 CPU |
| 平台 / 编译器 / 构建 | Ubuntu 22.04.5 / GCC 11.4 / RelWithDebInfo（沿用已有构建） |
| 线程 / 绑核 | 单线程 / taskset CPU 0，cpu_core P 核 |
| governor / Turbo | 准备快照 powersave / no_turbo=0，允许 Turbo |
| 接电 / 电源模式 / 实际持续频率 | 接电 / Ubuntu 平衡模式（用户确认） / 未记录 |
| 输入 / 后端 | 合成 100000×128、1000 查询、k=10、L2 / AVX2Four |
| warmup / runs / 额外检索 | 1 / 3 / 自算答案 1 轮、召回复算 1 轮 |
| cpu_core cycles:u | 46,519,812,043 |
| cpu_core instructions:u | 63,241,165,547 |
| IPC | 约 1.36 |
| perf elapsed / user / sys（s） | 13.515875130 / 13.468106 / 0.040985 |
| 程序输出 QPS / 平均延迟（ms） | 463.8 / 2.1562（正式三轮范围，perf 运行下数据） |
| 严格 Recall@10 | 1.000000，自身生成答案，不作为独立召回验收 |

perf 提示事件按 PMU 重组，cpu_atom 行 not counted、cpu_core 有效，符合本轮固定 P 核的配置。
只使用有效 cpu_core 的 instructions/cycles 求比，不把 atom 缺失计为零或累计进另一种 PMU。
1.36 是进程整个用户态采集范围的平均退休 IPC，不是 CPU 利用率、内核独占 IPC 或理论峰值占比。
该 IPC 为单轮结果，不能单独确认算力、带宽或分支瓶颈；本轮缓存与热点结果见下一节。


## 首轮缓存与热点结果

同目录、同一合成负载与 CPU 0。缓存统计开始于 2026-10-22 12:48:10，
环境沿用上述快照：i7-14650HX（16 核/24 线程）、GCC 11.4 RelWithDebInfo、
单线程固定 P 核、powersave、允许 Turbo；用户确认接电、Ubuntu 22.04 平衡模式。
平衡模式是系统电源策略，powersave 是 cpufreq governor，二者属于不同层次，不矛盾；
各次采集前没有独立重采频率快照，实际持续频率未记录。

| 缓存事件 | 有效 cpu_core 数值 |
|---|---:|
| cache-references:u | 4,817,257,222 |
| cache-misses:u | 2,829,051,563 |
| 事件比值 | 58.73% |
| elapsed（s） | 12.751395133 |

这是 PMU 通用缓存事件的比值，不能解释为“所有内存访问中 58.73% 都失败”，
也不能混合前一轮的 cycles 计算跨运行派生指标。不同缓存层、预取事件与通用别名语义需要区分。

`flat.perf.data` 约 20 MiB，使用 cycles:u、目标 199 Hz、DWARF 8192 字节栈采样；
`flat-report.txt` 报告约 2K 样本，Total Lost Samples=0（该报告口径）。

| 函数自身样本归属 | 比例 |
|---|---:|
| L2SqrAVX2Four | 93.95% |
| FlatIndex::Search | 4.40% |

这些是不同函数自身的样本归属，不能把 Search 4.40% 理解为包含其所有子调用的总占比。
调用栈表明，正式/预热 RunRound 与 SearchBatch 的检索都到达距离内核，符合整个进程覆盖范围。
内联 sub_ps 一行的样本比例也不等于减法指令独占耗时，指令定位会受优化、内联与采样偏移影响。

当前证据支持：该工作负载的 cycles 样本主要集中在距离扫描内核，
缓存通用事件未命中比例较高，应继续围绕距离内核的加载/归约分析，而非优先替换 top-k 堆。
但不能由此确认唯一瓶颈是访存、已饱和 DRAM 带宽或算术单元；这些需要更专门的事件或受控对照。
P1 已有初步运行时证据，Scalar/AVX2 的同条件 IPC 对照随后已完成，见下节；没有全量 SIFT1M 采样。


## 同条件距离内核 IPC 对照

2026-10-22 12:55，代理在沙盒内再次探针成功，paranoid=2。
顺序执行两组命令，无并行测量；原始文件位于
`data/p1-results/perf-OGEgW0/ipc-compare-teb3wh3d/`。
目录包含两份 Benchmark JSON、stat/控制台日志、精确参数数组、提交/工作树状态、CMake 缓存摘要和环境记录。

环境：Intel Core i7-14650HX（16 物理核/24 逻辑 CPU）、Ubuntu 22.04.5、GCC 11.4、RelWithDebInfo，
单线程固定 CPU 0（P 核），powersave、允许 Turbo；接电、Ubuntu 平衡模式沿用用户最新确认。
128 维、对齐、热缓存、直接调用；最短预热 0.2 秒、每次最短测量 0.5 秒，重复 5 次。
实际持续频率未测量，工具仍警告 CPU scaling enabled。

| 项目 | Scalar | AVX2Four |
|---|---:|---:|
| Benchmark CPU mean（ns） | 52.5321 | 5.1699 |
| CPU CV | 0.9306% | 0.0196% |
| 整进程 cpu_core cycles:u | 14,832,949,855 | 14,748,272,289 |
| 整进程 cpu_core instructions:u | 68,050,292,742 | 72,318,937,360 |
| 整进程 IPC | 4.59 | 4.90 |

按本轮 CPU mean，Scalar/AVX2Four 约为 10.16×。
IPC 增幅远小于耗时改善不矛盾：SIMD 单条指令可处理多个数据，IPC 不衡量每条指令完成多少维计算。
不要把 AVX2 的更高总 instructions 解读为单次计算更多指令：两组自动校准迭代数不同，
进程计数还包括初始化、校准和预热，绝对计数并非相同调用次数下的对照。

同机固定 P 核下，热缓存内核采集 IPC 高于整库扫描首轮的 1.36，结合距离函数的高样本占比与
缓存通用事件，支持将后续分析重点放在大数组扫描时的加载与计算交互。
这仍是不同负载/调用路径的对比（直接调用 vs Flat 间接调用），不能单独量化访存停顿、
证明 DRAM 已饱和或宣布达到 roofline。P1 初步 perf 采集与解释已完成，专门事件分析尚未开展。
