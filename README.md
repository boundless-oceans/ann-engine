# ann-engine

> **ANN** = *Approximate Nearest Neighbor*（近似最近邻）。
> 给定一个查询向量，在百万级向量库中找出**最接近**的那几个 ——
> **不保证精确，但快几个数量级**。这是向量检索、RAG、推荐系统的底层基础设施。

从零实现的高性能 ANN 检索引擎（C++20）。**手写** HNSW / IVF 索引与 PQ/SQ 量化，
不依赖 Faiss 等现成库，并与 Faiss 做性能对照。

> **状态：P1 开始，P0 原验收差异保留** —— Ubuntu 上 rel 构建的 27 个测试已通过，合成数据评测已跑通；
> san 构建的 27 个测试也已通过，日志无 ASan / UBSan 告警；全量 SIFT1M 严格 Recall@10=0.999300，
> 70 条差异已确认为同距离点的不同选择。保留严格指标与原验收标准，尚未达到 1.0。

---

## 目标

- 手写完整索引栈：**Flat（暴力 / ground truth）→ HNSW（图索引）→ IVF（倒排）+ SQ/PQ（量化）**
- 产出可验证的硬指标：**Recall@K · QPS · P50/P99 · 内存占用 · 构建时间**
- 与 **Faiss** 横向对比，给出**召回率-QPS 曲线**

## 技术要点

| 层 | 内容 |
|---|---|
| 基础 | 向量类型 · L2/IP 距离 · SIMD（标量 → AVX2 + 寄存器分块）· 内存对齐 |
| 索引 | Flat · HNSW（分层图）· IVF（倒排）· SQ / PQ（量化） |
| 存储 | 快照 + WAL · 崩溃恢复 · 内存池 |
| 服务 | HTTP / gRPC · 多线程搜索 · 写入串行化 |
| 评测 | fvecs 数据集 · Recall@K · QPS · 与 Faiss 对比 |

## 文档

> 设计文档在本地 `docs/` 下持续演进，目前被 Git 忽略，远程仓库中的链接可能不可用。

| 文档 | 内容 |
|---|---|
| [总设计](docs/design/design.md) | 三合一：**总设计**（目标 · 技术栈 · 架构 · 实现顺序 P0–P7 · 问题与对策）+ **架构图集 14 张** + **内存与 SIMD** |
| [类与接口](docs/design/interfaces.md) | 二合一：**UML 类图 + 分层细化 + 速查表** + **详细签名与成员布局** |
| [分布式扩展](docs/design/raft.md) | P7：Raft 共识层 / 状态机 / 传输 / 与索引的集成 |
| [技术决策记录](docs/decisions/) | ADR：每次关键取舍留一份（背景 / 候选 / 决定 / **代价**） |

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 工程骨架 + FlatIndex + 评测框架 | 🟡 rel/san 测试通过，基线已记录；严格 Recall=0.999300，差异已解释但原验收未达成 |
| P1 | SIMD 距离 + 内存池 | 🟡 运行时派发通过 rel/san 50 个测试及 128 维微基准；合成/SIFT1M 已复测；集成分支 rel/san 各 73 个通过，40 项微基准运行检查通过；初步 perf 分析已完成 |
| P2 | HNSW | 🟡 图存储、随机层数及搜索原语完成，rel/san 各 114 个通过；基础插入待做 |
| P3 | IVF + SQ/PQ | ⬜ 未开始 |
| P4 | 持久化 + 崩溃恢复 | ⬜ 未开始 |
| P5 | 服务化（gRPC / HTTP） | ⬜ 未开始 |
| P6 | 差异化（过滤检索 / GPU 批量距离 / 删除） | ⬜ 未开始 |
| P7 | Raft 分布式扩展（多副本一致） | ⬜ 未开始 |

## 目录结构

> 下面包含现有目录与后续规划，目录会随实现进度逐步建立。

```
ann-engine/
├── include/ann/     # 公共头文件
├── src/             # 实现
├── bench/           # 评测（数据集加载 · 指标）
├── tests/           # 单元测试（GTest）
├── tools/           # 服务端可执行
├── data/            # 数据集（不入库，见 data/README.md）
└── docs/            # 文档
    ├── design/      # 设计正文
    └── decisions/   # ADR：技术决策记录
```

## 开发环境

- **Ubuntu 22.04**
- GCC 11+（C++20；GCC 11/12 缺 `<format>`，需要时用 fmt）
- CMake 3.22+ · Ninja · ccache
- GoogleTest · Google Benchmark
- 构建步骤见 [总设计文档](docs/design/design.md)

## P0 手动验证

以下命令均在仓库根目录执行。首次配置通过 FetchContent 下载 GoogleTest，需联网。
默认源码目录为 `build-rel/_deps/googletest-src/`（san 对应 `build-san/`），不使用 `third-party/`。
所有构建预设默认最多 8 个并行任务；显式指定 `-j` 时使用 8 或更小，不使用单独的 `-j`。
新增了 12 个 `FlatIndex` 用例，连同已有测试共 27 个；用户已反馈 rel 全部通过。

### 1. 配置、构建与单元测试

```bash
cmake --preset rel
cmake --build --preset rel -j 4
./build-rel/bin/ann_tests --gtest_filter='FlatIndex.*'
ctest --preset rel
```

预期：FlatIndex 的 12 个测试通过，随后全部测试通过。
手算用例为 `(0,0)、(1,0)、(10,0)`，查询 `(0.9,0)`，top-2 ID 为 `[1,0]`，
距离约为 `[0.01,0.81]`。新增用例也验证最近点位于后段时的堆替换，避免只覆盖初始建堆。

### 2. ASan / UBSan 正确性检查

```bash
cmake --preset san
cmake --build --preset san -j 4
ctest --preset san
```

预期：全部测试通过，ASan / UBSan 无告警。本次未新增并发场景，TSan 并发验证随后续并发功能补齐。
性能数据只使用 `build-rel`，不使用 Sanitizer 构建。

已核验用户于 2026-10-18 执行的完整 `build-san/Testing/Temporary/LastTest.log`：
27 条测试记录全部通过，未发现 ASan / UBSan 告警；对应 CMake 缓存确认 Debug、ASan=ON、UBSan=ON。

### 3. 记录测量环境

```bash
mkdir -p data/p0-results
lscpu | tee data/p0-results/cpu.txt
free -h | tee data/p0-results/memory.txt
uname -a | tee data/p0-results/system.txt
g++ --version | tee data/p0-results/compiler.txt
cmake --version | tee data/p0-results/cmake.txt
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor | tee data/p0-results/governor.txt
```

记录 CPU 型号、物理核数/逻辑线程数、内存、频率 governor、Turbo 状态及是否绑核。
`lscpu` 是否展示 Turbo 状态取决于平台，未展示时需另行记录；governor 文件不可用时也记录为不可用。
编译器以 CMake 配置摘要中的实际版本为准；测量时保持电源模式一致并关闭高负载后台任务。
以下评测均为单线程查询，预热 1 轮、测量 3 轮。

### 4. 合成数据检查评测链路

```bash
./build-rel/bin/ann_bench --n 100000 --nq 100 --dim 128 --k 10 --metric l2 --warmup 1 --runs 3 | tee data/p0-results/synthetic.txt
```

预期：输出 Recall、QPS、P50/P95/P99、内存及构建时间。
合成数据由 FlatIndex 自己生成 ground truth，Recall 为 1.0 只能说明链路通了。

2026-10-19 用户手动执行上述参数的结果如下。这是合成数据基线，不作为 SIFT1M 的验收结果。

| 项目 | 本次记录 |
|---|---|
| CPU / 物理核数 / 逻辑线程数 | Intel Core i7-14650HX / 16 / 24 |
| 系统 / 编译器 / 构建配置 | Ubuntu 22.04.5 LTS / GCC 11.4.0 / RelWithDebInfo |
| 频率 governor | cpu0 为 powersave；其他逻辑 CPU 未记录 |
| Turbo / 电源模式 / 内存容量 | 未记录 / 未记录 / 未记录 |
| 查询线程数 / 绑核 | 1 / 未显式绑核，可能在不同核心间迁移 |
| 数据 / 度量 / k / seed | 合成数据（100000 × 128，100 queries）/ L2² / 10 / 42（默认）|
| warmup / runs | 1 / 3 |
| Recall@10 / QPS | 1.000000（自算答案）/ 174.8 |
| 平均 / P50 / P95 / P99 / 最大延迟（ms） | 5.7198 / 5.5939 / 6.4575 / 6.7573 / 7.1598 |
| 索引内存 / 构建时间 | 48.8 MiB / 0.019 s |

这是未绑核、powersave governor 下的一次测量。后续优化对比需要保持相同测量条件；
若改用固定核心或其他电源策略，应重新测量对照基线。

### 5. 完整 SIFT1M 正式验收

按 [数据集说明](data/README.md) 准备三个文件，可先检查其字节数：

```bash
wc -c data/sift_base.fvecs data/sift_query.fvecs data/sift_groundtruth.ivecs
```

标准 SIFT1M 的字节数分别为 `516000000`、`5160000`、`4040000`，包含每条记录的维度字段。
文件大小只是初步检查，还需确认评测输出为 `base: 1000000 x 128`、`queries: 10000`。

```bash
./build-rel/bin/ann_bench --data data --k 10 --metric l2 --warmup 1 --runs 3 | tee data/p0-results/sift1m.txt
```

本命令对全量查询执行预热、测量及额外的召回校验，标量 Flat 扫描耗时可能较长。
`--data` 模式使用文件中的查询条数，`--nq` 不会减少查询数量。
正式验收不加 `--max-base`：截断 base 后会改用自算答案。
SIFT 官方答案用于 L2 验收，不用于内积验收。

P0 通过条件：单元测试及 ASan / UBSan 全部通过，完整 SIFT1M 上 `Recall@10 == 1.0`，
并记录 SIFT1M 基线与测量环境。2026-10-19 用户已完成全量评测，输出保存于
`data/p0-results/sift1m.txt`（本地文件，不入库）。本次结果如下，**尚未通过 P0 召回率验收**。

| 项目 | 基线记录 |
|---|---|
| CPU / 物理核数 / 逻辑线程数 / 内存 | Intel Core i7-14650HX / 16 / 24 / 内存容量未记录 |
| governor / Turbo / 绑核 / 电源模式 | 前次 cpu0 为 powersave，本轮未重新采样 / 未记录 / 未显式绑核 / 未记录 |
| 系统 / 实际编译器 / 构建配置 | Ubuntu 22.04.5 LTS / GCC 11.4.0 / RelWithDebInfo（沿用此前构建与环境）|
| 数据 / 度量 / k | SIFT1M（1M × 128，10K queries）/ L2² / 10 |
| 查询线程数 / warmup / runs | 1 / 1 / 3 |
| Recall@10 / QPS | **0.999300** / 17.8 |
| 平均 / P50 / P95 / P99 / 最大延迟（ms） | 56.1607 / 56.3556 / 58.5170 / 67.3618 / 93.0090 |
| 索引内存 / 构建时间（s） | 488.3 MiB / 0.198 |

已通过下一节的诊断工具定位：10000 条查询中有 70 条各缺少一个官方 top-10 ID，
由另一个相同距离的点替代。全部 70 条差异查询的排序后 double 距离序列与官方答案、独立参考完全相等，
float 返回距离也与 double 重算结果一致。本次差异没有发现更远点替代更近点的证据。
例如官方 ID 266710 与返回 ID 202432 的 L2² 均为 48069。

用户决定保留严格按 ID 匹配的 Recall，不增加替代指标，不为追平官方平局选择修改检索逻辑。
`Recall@10 == 1.0` 的原验收标准保持不变，本次未达成；后续结果继续如实报告严格 Recall。
Sanitizer 验证已通过，召回差异分析已完成，原验收未达成事项继续保留。

### 6. 诊断 SIFT1M 召回差异

新增 `ann_recall_diag`，固定检查 L2 top-10，不截断 base。
工具已由用户手动编译并完成全量运行；复现命令如下：

```bash
cmake --build --preset rel --target ann_recall_diag -j 8
mkdir -p data/p0-results
./build-rel/bin/ann_recall_diag data > data/p0-results/recall-diag.txt
```

现有 Ninja 构建会在检测到 CMakeLists.txt 变化后重新生成；依赖源码位置沿用构建缓存。
GoogleTest 已改为使用项目内 `build-deps/googletest-1.15.2`，避免 `/tmp` 被清理导致重新配置失败。
若旧缓存仍指向 `/tmp`，在源码已到位的前提下执行
`cmake --preset rel -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build-deps/googletest-1.15.2"`。
进度每 100 条查询输出到终端，详细结果写入文件。
只做一轮查询，不预热或跑性能轮次；对每个 ID 不一致的查询，额外独立扫描全部 base，
用 double 距离和标准库部分选择生成参考 top-10。
输出包括缺失/额外 ID、float 返回距离、double 重算距离、官方与参考结果。

```bash
tail -n 10 data/p0-results/recall-diag.txt
```

`strict_recall` 仍严格按 ID 匹配；`both_distance_equivalent_to_reference` 表示双方距离序列
都与参考结果匹配的查询数（容差 `1e-9 * max(1, |a|, |b|)`），仅用于分析。
参考结果按距离再按 ID 排序，并不代表官方答案采用相同的平局规则。
出现 `unresolved_queries` 时应查看明细；诊断工具正常结束不等于原 Recall 验收通过。

本轮汇总：`strict_recall=0.999300000`、`missing_ids=70`、`mismatched_queries=70`、
`both_distance_equivalent_to_reference=70`、`unresolved_queries=0`。
原始明细在 `data/p0-results/recall-diag.txt`（本地文件，不入库）；另已逐条核对输出，
70 条差异查询的距离一致性均为完全相等，并非仅依赖诊断分类容差。

## P1 距离微基准

已新增可选目标 `ann_distance_bench`，使用 Google Benchmark v1.9.1 测量现有标量 L2² 与负内积，
覆盖 16、127、128、129、960 维；用户已完成编译、运行检查与 128 维正式测量。
依赖准备、构建、检查、测量环境及运行命令见 [微基准说明](bench/README.md)。
采用相同配置比较后续 AVX2 单累加器与四累加器版本；单对热缓存的 ns 结果不能直接代替检索 QPS。

首次标量 CPU 耗时均值：L2² 为 **57.7895 ns**，负内积为 **48.7095 ns**，CV 分别为
1.8742% / 1.6883%。测量环境为 i7-14650HX（16 核/24 线程）、Ubuntu 22.04.5、GCC 11.4、
RelWithDebInfo，cpu0 powersave、允许 Turbo，单线程未显式绑核；128 维热缓存输入，最短预热
0.2 秒、每次最短测量 0.5 秒、重复 5 次。完整记录与限制见微基准说明。

新增 AVX2 + FMA 单累加器 L2²/负内积、8 个测试与 10 个微基准条目，已由用户编译并完成 rel/san
全部 35 个测试及 128 维正式测量，完整日志无 Sanitizer 告警。
支持任意维度和非 32 字节对齐的输入；高阶指令通过函数属性隔离，测试与微基准先检查 CPU 特性。
单累加器验收时 `SelectDistFn` 仍选择标量；该阶段手动检查及同轮测量命令见微基准说明第 6 节。

2026-10-19 同轮 CPU 均值对比：L2² Scalar 63.8192 ns → AVX2Single 6.8118 ns（9.37×），
负内积 Scalar 62.2633 ns → AVX2Single 5.9057 ns（10.54×）。环境为 i7-14650HX（16 核/24 线程）、
Ubuntu 22.04.5、GCC 11.4、RelWithDebInfo，cpu0 powersave、允许 Turbo；单线程未绑核，
128 维热缓存，最短预热 0.2 秒、每次最短测量 0.5 秒、重复 5 次。
这是单对距离微基准结果，不代表 Flat QPS 同比例提升；详细波动与环境限制见微基准说明。

四累加器已从功能子分支 `feat/p1-avx2-four-accumulators` 合回 P1，保留单累加器入口作为对照。
新增 6 个测试和 10 个微基准条目（总计 41 个测试、注册 30 个微基准），已完成编译与 rel/san
全部 41 个测试，完整日志无 Sanitizer 告警；128 维三版本同轮测量已完成。
rel/san 检查及三版本同轮测量命令见 [微基准说明第 7 节](bench/README.md#7-四累加器子分支与手动验证)。

四累加器同轮 CPU 均值：L2² 为 5.1599 ns，负内积为 5.2830 ns；相对单累加器分别为
1.35× / 1.10×，相对本轮 Scalar 为 12.34× / 11.00×。环境沿用 i7-14650HX（16 核/24 线程）、
GCC 11.4、RelWithDebInfo，单线程未绑核，128 维热缓存，预热最短 0.2 秒、测量最短 0.5 秒、重复 5 次。
本轮 Benchmark 提示频率调节开启，但未保存独立 governor/Turbo 快照，前次 powersave/允许 Turbo
不作为本轮已确认状态。原始 JSON、完整三版本结果和环境缺项见微基准说明第 7 节。

此前在 `feat/p1-distance-dispatch` 实现运行时派发：通过缓存的 CPU 特性，在同时支持 AVX2 与 FMA 时
选择四累加器，否则回退标量。`FlatIndex` 构造时保存函数指针，`Search` 不做 CPU 检测。
新增 9 个派发测试与 10 个间接调用微基准（总计 50 个测试、注册 40 个微基准）。用户已完成 rel/san 各 50 个测试，已核对完整日志及 Sanitizer 开关，无相关告警；128 维同轮测量已确认 Dispatch 使用 AVX2Four，端到端合成与全量 SIFT1M 复测已完成，原始输出及结果见微基准说明第 8 节。
手动验证命令、测量条件与严格 Recall 复核步骤见 [微基准说明第 8 节](bench/README.md#8-运行时-cpu-派发与手动验证)。

已从 `feat/p1-aligned-memory` 合入 32 字节对齐连续存储：新增 `AlignedAllocator` 与 `AlignedVector<T>`，
Flat 仍按 n×dim 紧密排列，AVX2 保留 loadu，查询仍只读且零分配。新增 13 个测试，代理已在沙盒完成
GCC 11.4 rel/san 各 63 个测试；本轮 san 因沙盒 LeakSanitizer 环境错误临时关闭泄漏扫描，
ASan/UBSan 无告警，用户随后已在普通终端完成包含这些用例的默认 san 检查。没有新增性能结论或重跑全量 SIFT1M；
对齐存储阶段的实现取舍、复现命令与面试点见本地 [对齐内存笔记](docs/learn/aligned-memory.md)。

当前 `feat/p1-build-arena` 已实现构建期内存池：缓存上游保留单调资源归还的块，Reset 清零用户申请计数、
保留容量供后续构建复用。Reset 前必须先销毁使用池内存的对象和容器，池非线程安全，不进入 Search。
新增 10 个测试，GCC 11.4 rel/san 各 73 个通过；本轮沙盒 san 临时关闭 LeakSanitizer，ASan/UBSan 无告警，
用户已报告普通终端默认 san 的 73 个测试全部通过。当前仅交付基础设施，尚未接入索引构建，无新性能结论；下一步为 perf 分析。
取舍、生命周期示例及面试要点见本地 [BuildArena 笔记](docs/learn/build-arena.md)。

perf 已进入采集准备：工具可用，但初始探针在代理及用户终端均被拒绝；用户处理后普通终端硬件探针成功，代理环境仍受限。
已整理 [perf 采集方案](bench/perf.md)，固定 P 核的 Flat 合成负载已采集首轮 IPC≈1.36；缓存与采样已完成首轮：主要 cycles 样本落在 L2SqrAVX2Four，结果、环境与归因边界见采集方案；Scalar/AVX2 的固定 P 核 IPC 对照也已完成，原始数据与解释见采集方案；不宣称已达到 roofline。

2026-10-22 已在 P1 集成提交 `dd1027c` 完成 rel/san 各 73 个测试及 40 项微基准运行检查，
本轮沙盒 san 仍临时关闭泄漏扫描，用户此前普通终端默认 san 通过记录单独保留。
阶段交付、性能范围和未验证项见 [P1 集成验证记录](docs/validation/p1.md)，接下来准备 P2 图存储。

交付时的面试追问与回答要点汇总在本地 [面试讲点](docs/learn/interview-notes.md)，
与原 [学习记录](docs/learn/learning-log.md) 分开维护；`docs/learn/` 当前被 Git 忽略，不入库。

## P2：HNSW 图存储起步

图存储已在 `feat/p2-hnsw-graph-storage` 完成并合回 P2；随机层数也已合回 P2，当前按用户选择直接在 `feat/p2-hnsw` 推进相关功能，减少短分支。
新增私有组件 `src/hnsw_graph.h/.cpp`：向量连续对齐存储，L0 定长邻居槽位，上层按节点实际层数分配，
节点引用和上层偏移使用 32 位整数，空槽位用 `kInvalidIdx` 标记。
提供显式层数追加、只读视图、入口维护和有向邻接表替换；不在存储层自动做双向连接或剪枝。
新增 13 个测试，包含二维十点小图、300 次追加、层间隔离、非法邻居更新及 32 位偏移上限。
GCC 11.4 rel/san 各 86 个测试通过；本轮沙盒 san 临时关闭泄漏扫描，ASan/UBSan 无告警。
用户随后已在普通终端通过默认 san 的全部114个测试，包含此前图存储和随机层数用例。当前没有可查询的 HNSWIndex，尚无 HNSW Recall/QPS 结果；图存储阶段验证记录如上。
本地讲解与面试要点见 [图存储笔记](docs/learn/hnsw-graph-storage.md)。

新增内部 `HNSWLevelGenerator`（src/hnsw_level.h/.cpp），按 `floor(-ln(U)/ln(M))` 生成最高层，
每实例独立 mt19937_64，默认种子 42；显式映射严格开区间 U，安全限制 uint8_t 层范围。
新增 10 个测试，总计 rel/san 各 96 个通过；本轮沙盒 san 关闭泄漏扫描、ASan/UBSan 无告警，
用户随后已在普通终端通过默认 san 的全部114个测试，包含此前图存储和随机层数用例。M=2/4/16/32 各 100 万样本的尾分布与均值检查通过，
另完成生成层数与 5000 节点图存储的集成检查；尚未实现自动连接、插入或搜索。
原理、样本统计与面试要点见本地 [随机层数笔记](docs/learn/hnsw-random-level.md)。该阶段后续搜索原语状态见下段。

新增私有 `src/hnsw_search.h/.cpp`：严格改善的单层贪心下降、双堆单层 ef 搜索，
通过 const 图引用只读访问，临时堆和访问表由调用方独占工作区持有。工作区提前准备，
搜索不扩容；代价是每次清零 N 字节访问表和保守 O(N) 候选容量。结果最多 ef 个，
可达节点不足时可以更少，按距离/ID 排序；该规则不改变 P0 严格 Recall 定义。
新增 18 个搜索测试，GCC 11.4 rel/san 各 114 个通过，本轮沙盒 san 关闭泄漏扫描、ASan/UBSan 无告警，
用户已在普通终端通过默认 san 的114个测试。尚无完整 HNSWIndex、自动插入或 HNSW Recall/QPS 结果；
下一块接入基础插入与完整索引查询。原理与面试点见本地 [分层搜索笔记](docs/learn/hnsw-layer-search.md)。

## 数据集

SIFT1M（1M × 128d）。**不入库**（体积大），获取方式见 [data/README.md](data/README.md)。

## 命名与约定

- 命名空间 `ann`
- 索引统一实现 `Index` 抽象接口（见总设计文档 §2.2）
- **索引内部一律用下标（`uint32_t`）而非指针** —— 既利于 cache，也让持久化几乎免费
