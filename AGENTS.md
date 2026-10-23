# AGENTS.md

> 给编码代理（Codex CLI / Claude Code / DSH 等）的项目指引。
> **若 `docs/design/` 为空**，说明设计文档还没从本地拷入 —— 此时先向用户索要。

---

## 这个仓库是什么

**ann-engine** —— 从零手写的高性能 ANN（近似最近邻）检索引擎，C++20，不依赖 Faiss。

> **P1/P2 已由用户提交并合回 main（045f26b），当前 P3 功能分支推进未量化 IVF；保留 P0 原验收未达成记录。** P0 实现已写完（`src/distance.cpp` · `src/flat.cpp`），
> 已由用户在 Ubuntu 上完成 rel 构建，27 个测试全部通过，合成数据评测已跑通；
> 全量 SIFT1M 已测得 Recall@10=0.999300，尚未达到 1.0；
> san 构建的 27 个测试也已通过，日志无 ASan / UBSan 告警；70 条召回差异已确认为同距离点的不同选择。
> 用户决定保留严格 Recall，不增加替代指标、不为追平官方平局选择修改检索逻辑；原验收标准仍未达成。

## 先读这些（按顺序）

| # | 文件 | 内容 |
|---|---|---|
| 1 | `README.md`（仓库根） | **项目介绍**：目标 · 技术要点 · 路线图 · 目录结构 · 开发环境 · 命名约定 |
| 2 | `docs/design/design.md` | **总设计**：目标与量化验收标准 · 技术栈 · 架构 · **实现顺序 P0–P7** · 问题与对策；含 14 张架构图 + 内存布局/SIMD 底层分析 |
| 3 | `docs/design/interfaces.md` | **类与接口**：UML 类图 + 分层细化 + 详细签名与成员布局 ← **动手前必读** |
| 4 | `docs/design/raft.md` | P7 分布式扩展（Raft 共识层 / 状态机 / 传输 / 与索引集成） |
| 5 | `docs/decisions/` | 技术决策记录（ADR）：每次关键取舍的背景 / 候选 / 决定 / **代价** |

## 硬约束

- **语言 C++20**，编译器 **GCC 11+**（开发机是 Ubuntu 22.04 自带的 GCC 11.4）
  - GCC 11 的 libstdc++ 缺 `<format>`（GCC 13 才有）→ 需要时用 `fmt` 替代
  - 其余用到的 C++20 特性在 GCC 11 上均可用：`concepts` · `<ranges>` · `std::span`
    · `std::pmr` · `<bit>` · `std::atomic::wait`
- 命名空间 `ann`；头文件 `include/ann/`，实现 `src/`
- **头文件同时使用 `#pragma once` 与 `#ifndef` 包含卫士**（双重保险）
  - 卫士宏命名统一为 `ANN_<路径>_H`：`ann/types.h` → `ANN_TYPES_H`，
    `bench/dataset.h` → `ANN_BENCH_DATASET_H`
  - `#pragma once` 走快路径，包含卫士在同文件经不同路径被包含时兜底
- **内部引用一律用 `uint32_t` 下标，绝不用指针**
  - 这一条同时关系到 cache 性能、持久化免重定位、以及未来分片
- **`Index::Search` 必须是 `const` 且不修改任何成员** —— 这是并发的前提；
  需要临时 buffer 时放栈上或由调用方传入，不要放成员变量
- **核心实现全部手写**：不引入 Faiss / hnswlib / Annoy / Eigen
  - Faiss 只作为 **Python 侧**的对比基线，不链接进 C++ 项目
- 数据集**不入库**（`data/` 被忽略），获取方式见 `data/README.md`

## 当前进度与下一步

路线图见 `README.md`（仓库根）。

**已完成**：

- 构建系统（CMake + 预设）
- 接口声明（`include/ann/`）
- 评测框架（`bench/`）
- 距离函数实现 + 单元测试（`src/distance.cpp` · `tests/distance_test.cpp`，9 个用例）
- `FlatIndex` 实现（`src/flat.cpp`，top-k 用大小为 k 的最大堆，`Search` 零分配）
- `tests/flat_test.cpp`（12 个用例），已接入 CMake
- Ubuntu 上 rel 构建及全部 27 个测试通过（用户手动执行）
- Debug + ASan / UBSan 构建的全部 27 个测试通过，已核对完整 LastTest.log 和 CMakeCache.txt，日志无 Sanitizer 告警
- 合成数据基线已记录在 README（环境与参数随结果记录）
- 完整 SIFT1M 评测已运行，结果及测量环境信息见 README；原始输出在 `data/p0-results/sift1m.txt`
- 召回差异诊断工具 `bench/recall_diag.cpp` 已由用户编译并完成全量运行；70 条差异均经独立 double 参考复核，距离序列完全一致，未解释差异为 0

**P0 剩余**：

- 原严格 Recall@10=1.0 验收未达成（实测 0.999300），原因已解释并保留记录，不再当作未知原因重复排查
- 后续测量补齐本轮频率状态等环境信息，保持可复现的基线记录

**验收标准**：`加载 SIFT1M → FlatIndex → Recall@10 == 1.0`，并记录一个基线 QPS 数字。

**P1 完成记录**：

- `feat/p1-simd-and-memory` 作为 P1 集成分支；具体功能在子分支开发并验证后合回。P1 已由用户合回 main，当前位于 `feat/p2-hnsw`，代理编写代码；用户已于 2026-10-20 授权代理自主执行沙盒内安全的构建、测试与测量，Git 写操作仍由用户执行
- 可选 Google Benchmark v1.9.1 距离微基准（`bench/distance_bench.cpp`、目标 `ann_distance_bench`）已由用户编译并完成 10 个用例的运行检查、128 维标量正式测量
- 开关 `ANN_BUILD_DISTANCE_BENCHMARKS` 默认 OFF；启用、依赖准备及手动命令见 `bench/README.md`
- 首次标量缓存内基线与环境已记录在 `bench/README.md`（原始 JSON 在 `data/p1-results/scalar-128.json`）；后续接入 AVX2 时需同轮重测 Scalar，不使用跨环境结果直接计算加速比
- AVX2 + FMA 单累加器已编写（`src/distance_avx2.cpp`），支持 loadu 和标量尾部，函数级 ISA 隔离；单累加器验收阶段 `SelectDistFn` 尚未接入 SIMD
- 新增 `tests/distance_avx2_test.cpp`（8 个测试，总计 35 个）和 AVX2Single 微基准条目（总计注册 20 个）；用户已完成编译、rel/san 全部 35 个测试及 128 维同轮对比，完整日志无 Sanitizer 告警，结果与环境见 `bench/README.md` 第 6 节
- 微基准兼容性修复：用显式布尔模板标记区分 SIMD，避免 GCC 11 不接受函数地址常量比较；DoNotOptimize 使用非 const 结果变量
- 四累加器 L2/IP 入口（`L2SqrAVX2Four` / `InnerProductAVX2Four`）及 6 个测试、10 个微基准条目已由用户编译并完成 rel/san 全部 41 个测试，完整日志无 Sanitizer 告警；总计注册 30 个微基准。128 维三版本同轮测量已完成，结果、环境及 governor/Turbo 本轮未采样的缺项见 `bench/README.md` 第 7 节
- 面试追问与回答要点已按用户要求汇总至 `docs/learn/interview-notes.md`，原 `docs/learn/learning-log.md` 保留；该目录当前被 Git 忽略
- 四累加器功能子分支已由用户提交并合回 P1 集成分支
- 运行时派发已编写：`CpuFeatures::Get()` 缓存实际 CPU 特性，`SelectDistFn` 在 AVX2/FMA 同时可用时选择四累加器，否则回退标量；`SelectDistFnScalar` 保留显式标量选择。`FlatIndex` 继续在构造时缓存指针，`Search` 不检测 CPU、不修改成员
- 新增 9 个派发测试，用户已完成 rel/san 各 50 个测试，已核对完整 LastTest.log 和 san 开关，日志无 Sanitizer 告警；模拟特性只比较地址，不执行不受支持的指令。新增 10 个 Dispatch 间接调用微基准（共注册 40 个）；128 维同轮测量确认后端 AVX2Four，结果与环境见 `bench/README.md` 第 8 节。派发后的合成端到端复测已完成（原始输出 `data/p1-results/dispatch-synthetic.txt`，确认 AVX2Four；自算答案不替代独立召回验证）；全量 SIFT1M 已完成，严格 Recall@10=0.999300，与 P0 汇总值一致；原 1.0 目标仍未达成，结果与环境缺项见 `bench/README.md` 第 8 节
- 派发设计取舍见 `docs/decisions/0004-runtime-distance-dispatch.md`，学习与面试要点见 `docs/learn/distance-dispatch.md`；运行时派发已由用户提交合回 P1；对齐存储已由用户提交并合回 P1，当前推进构建期内存池，之后为 perf 分析
- 对齐存储已实现：新增 `memory_pool.h/.cpp`，AlignedAllocator 使用 posix_memalign/free，AlignedVector 以标准容器加自定义分配器管理生命周期；Flat 紧密布局与 loadu 保持不变，MemoryUsage 统计申请容量。新增 13 个测试，总计 rel/san 各 63 个通过；代理在沙盒执行，ccache 临时禁用，san 因 LeakSanitizer 环境错误设置 detect_leaks=0，ASan/UBSan 无告警，用户随后在普通终端完成默认 san 验证（73 个测试包含这些用例）。本块未重跑全量 SIFT1M，无新性能结论。取舍见 ADR-0005，学习笔记 `docs/learn/aligned-memory.md`；对齐存储阶段尚未实现 BuildArena，后续状态见下条
- BuildArena 已按用户确认的原语义实现：Reset 保留全部缓存容量，单调资源通过缓存上游复用整块；计数前端统计本轮累计用户申请字节，deallocate 不减少，Reset 清零。initial_bytes 为惰性首块提示，CapacityBytes 统计缓存块申请量。池不可复制/移动、非线程安全，Reset 前必须先销毁使用池的容器（clear 不够），Resource 地址稳定。新增 10 个测试，代理完成 GCC 11.4 rel/san 各 73 个通过，san 临时 detect_leaks=0、ASan/UBSan 无告警，用户已报告普通终端默认 san 的 73 个测试全部通过；未接入索引构建、无新性能数字。取舍见 ADR-0006，学习讲解见 `docs/learn/build-arena.md`。BuildArena 和 perf 文档已由用户提交并合回 P1；集成验证见下条
- perf 准备（2026-10-22）：已安装 perf 6.8.12，内核 6.8.0-138-generic；perf_event_paranoid=4，代理沙盒及用户普通终端 task-clock 探针均被权限拒绝，未修改系统策略，无 IPC/cache-miss/热点实测。sysfs cpu_core=0-15、cpu_atom=16-23，当前允许 CPU 0-23；采集方案使用 CPU 0。命令、统计范围和待采集项见 `bench/perf.md`，面试要点见 `docs/learn/perf-analysis.md`。用户普通终端处理权限后已成功采集 true 的用户态硬件事件（cpu_atom，非项目测量），代理复查仍读到 paranoid=4 且探针被拒绝；用户固定 CPU 0 的 Flat 合成负载已采到有效 cpu_core IPC≈1.36（cycles=46,519,812,043，instructions=63,241,165,547），原始目录 data/p1-results/perf-OGEgW0；环境与范围见 bench/perf.md。同负载缓存事件比值58.73%，cycles样本自身归属 L2SqrAVX2Four 93.95%、Search 4.40%，报告无丢样本；用户确认接电/Ubuntu 平衡模式。已有初步热点证据，不据此宣称 DRAM 带宽饱和；代理随后沙盒探针成功并顺序完成 Scalar/AVX2Four 的 L2/128 IPC 对照（CPU 0、powersave、允许 Turbo、接电/平衡、单线程、预热0.2s、最短0.5s、5次）：CPU mean 52.5321/5.1699ns，同轮10.16倍；整进程IPC4.59/4.90。结果目录 data/p1-results/perf-OGEgW0/ipc-compare-teb3wh3d，P1 初步 perf 采集/解释已完成，不宣称带宽饱和或达到roofline，详见 bench/perf.md
- P1 集成回归（2026-10-22，dd1027c）：rel/san 全量构建及各 73 个测试通过，40 项距离微基准列表与 dry run 通过；本轮 san 临时 detect_leaks=0，无 ASan/UBSan 告警，用户此前默认 san 通过记录分开保留。只读确认 perf_event_paranoid=4。本轮无新性能测量/全量 SIFT1M 重跑。阶段验证与局限见 `docs/validation/p1.md`，收尾文档已由用户提交并将 P1 合回 main，P2 分支已建立
- 严格 Recall 的原有决定继续保持；构建并行不超过 8 个任务

## P2 当前工作

- P2 集成分支为 `feat/p2-hnsw`，当前位于 `feat/p2-hnsw`（图存储与随机层数均已由用户合回；用户2026-10-23明确选择直接在此分支推进相关功能，减少短分支），工作区起始干净
- 已实现内部 `ann::detail::HNSWGraphStorage`（src/hnsw_graph.h/.cpp），保持设计中的 SoA：AlignedVector<float> 向量、L0 定长槽位、上层 L×M 按节点分配、uint32_t 上层偏移与 uint8_t 层元数据；用 kInvalidIdx 哨兵前缀表示实际邻居，不额外维护度数数组
- 显式给层数 Append，入口只在新最高层出现时更新；SetNeighbors 只替换有向表，验证邻居存在/同层/无自环/重复/越界/超容量。图存储组件不负责搜索、随机层数、自动双向连接或剪枝，该存储阶段尚未公开可查询 HNSWIndex，当前集成状态见后续记录
- 新增 tests/hnsw_graph_test.cpp 的 13 个用例，含二维十点图、扩容稳定 ID、层间隔离、上层偏移范围等；代理完成 GCC 11.4 rel/san 各 86 个测试，本轮 san 临时 detect_leaks=0、ASan/UBSan 无告警，用户随后已在普通终端通过默认 san 的全部114个测试，包含本块用例；P1 的默认 73 项通过不代替本块 86 项验证
- 返回的只读 span 在扩容后可能失效，图边只存 ID/偏移；MemoryUsage 统计实际申请容量。Append 分配失败回滚逻辑状态，但可能保留容量并使旧视图失效；未注入真实 OOM 覆盖所有分配失败点
- 取舍见 ADR-0007，学习与面试要点见 docs/learn/hnsw-graph-storage.md。该块后续为随机层数（状态见下条），再按设计推进搜索、基础插入、启发式剪枝；尚无 HNSW Recall/QPS 结论

- 随机层数已实现为内部 HNSWLevelGenerator（src/hnsw_level.h/.cpp）：每实例 mt19937_64、默认种子42，M≥2，缓存1/ln(M)，按 floor(-ln(U)/ln(M)) 生成 uint8_t 最高层。高52位中点映射保证0<U<1，极值纯转换饱和到255；不宣称跨libm绝对一致或全图复现
- 新增10个随机层数测试：边界、同/异种子、状态独立、M=2/4/16/32各百万样本尾分布/均值、5000节点与图存储集成。代理完成 rel/san 各96个测试，本轮san临时detect_leaks=0、ASan/UBSan无告警，用户随后已在普通终端通过默认 san 的全部114个测试，包含此前图存储和随机层数用例。学习笔记 docs/learn/hnsw-random-level.md，取舍 ADR-0008
- 随机层数之后接入搜索原语（见下条）；当前无自动插入、选邻或HNSW Recall/QPS数据。分布公式：P(L≥l)=M^(-l)，E[L]=1/(M−1)；1/ln(M)是缩放因子而非每层晋升概率

- 搜索原语已实现：src/hnsw_search.h/.cpp 内部 HNSWGreedyDescend、HNSWSearchLayer、HNSWSearchWorkspace。只读图，调用方预备并独占工作区；候选最小堆、结果最大堆、字节访问表去重；每节点至多入候选一次，提前预留N个候选及min(N,ef)结果槽位，搜索不扩容，每次清零N字节标记。输出最多ef个，距离/ID排序，可达节点不足时可更少；不改变P0严格Recall
- 新增18个搜索测试，代理完成GCC11.4 rel/san各114个通过（本轮detect_leaks=0，ASan/UBSan无告警，用户随后已报告普通终端默认 san 的114个测试全部通过）。覆盖double参考、层隔离、ef桥接、环去重、断图、平局、非法参数、工作区复用/隔离、图只读、非有限距离和异常后复用；尚无真实并发/TSan或全局分配器拦截验证
- 取舍见ADR-0009，讲解 docs/learn/hnsw-layer-search.md；工作区O(N)存储/清零有成本，代际标记等优化留待测量。公开Index签名未改，完整HNSWIndex工作区接入留到索引集成阶段。该搜索阶段尚无HNSW Recall/QPS测量；后续集成状态见下条

### 2026-10-23 基础索引集成与接续点

- 已在当前 `feat/p2-hnsw` 实现公开 `include/ann/hnsw.h`、`src/hnsw.cpp`，组合图存储、随机层数和搜索原语；新增 `tests/hnsw_test.cpp` 的13个测试。用户已分三次提交：e17fb6a 实现与测试、27bde47 评测入口、52abb07 文档；Git 写操作仍由用户执行。
- Add 采用最近 M 连接及反向最近邻裁剪（L0 容量 Mmax0，上层 M），尚无多样性启发式，不保证所有边互反或全图连通。每点规划全部改边后再 Append/提交；当前点规划失败恢复 RNG，已完成批量前缀保留。BuildContext/BuildArena 在一批 Add 内复用，批次结束释放；未注入所有 OOM 点。
- 用户明确确认：先用函数内局部工作区跑通标准 Search 基线，不新增公开工作区接口。Search 保持 const 且不修改成员，每次有分配和 O(N) 访问表初始化成本，不能称为零分配。SetEfSearch 原子更新，Params 返回构造快照；Add 不可与查询/统计并发。
- 代理完成 GCC11.4 rel/san 各127个测试；本轮 san 使用 detect_leaks=0，无 ASan/UBSan 告警。用户随后已报告普通终端默认 san 的127项全部通过（5.78秒）；这是用户报告，代理未另行复核该次完整日志。小数据 L2/负内积都与独立 Flat 比较。
- ann_bench 新增 --index hnsw、--m、--ef-construction、--ef-search，默认仍 Flat；HNSW 合成数据使用独立 Flat 参考，严格 ID Recall 不变；--data 官方 SIFT 答案只允许 L2。
- 小规模基线原始目录 `data/p2-results/nearest-m-final-us017dob`；完整参数、环境和结果见 `bench/hnsw.md`。该合成5000×16/100查询基线不是全量SIFT1M验收，无P2最终达标结论。MemoryUsage 排除局部构建/查询工作区，不是峰值RSS。
- 取舍 ADR-0010，学习及面试表 `docs/learn/hnsw-baseline-index.md`。基础索引已提交，后续先完成较大数据最近M基线，再推进启发式剪枝。未做全量 SIFT1M、Faiss 对照、真实并发/TSan；不要重复排查 P0 已解释的平局差异。

### 2026-10-23 最近 M 真实数据子集基线

- 已在干净提交52abb07上顺序完成 HNSW/Flat 的 SIFT前100000条、128维、10000查询对照，L2、k10；因截断库，参考答案由该子集上的 Flat 重算，严格ID定义不变。HNSW M16/Mmax0=32、efC200/efS64、seed42。完整结果与环境见 bench/hnsw.md，原始目录 data/p2-results/nearest-m-sift100k-a1nvea4c。
- 环境为 i7-14650HX（16核/24逻辑CPU）、Ubuntu22.04/GCC11.4/RelWithDebInfo，单线程固定CPU0、powersave、允许Turbo、预热1轮/正式3轮；开始/完成后均读到接电，电源模式与实际频率未确认。结果为子集基线，不是全量SIFT1M或P2最终验收；核心代码本轮未改，不重复构建/测试。
- 下一块：按已有设计接入多样性启发式剪枝并补针对性测试，保留当前最近M版本作为同轮对照；尚未进行全量SIFT1M、Faiss或真实并发验证。docs/design和docs/decisions暂保留，用户希望收尾时评估整理后删除，当前未删除。

### 2026-10-23 多样性选邻（已提交）

- 当前仍在 feat/p2-hnsw，新增私有src/hnsw_neighbors.h与tests/hnsw_neighbors_test.cpp；HNSWSelectNeighbors用于新点选邻和反向超限裁剪，公开接口/SoA布局不变。候选按中心距离/ID排序且唯一，超容量时若到某个已选邻居距离严格小于到中心距离则拒绝；等距本身不拒绝。候选未超容量保留全部，不扩展/回填，允许输出少于M，不保证双向/全局连通；负内积不宣称度量几何保证。
- 两个独立输出缓冲放在批量BuildContext中；距离回调解析尚未Append的新点输入。仍先规划全部改边再Append/提交，Search未加共享状态；用户已提交ad30a18实现/测试和2afb07f验证文档，Git写操作仍由用户执行。
- 新增10项测试，代理完成GCC11.4 rel/san各137项通过，san临时detect_leaks=0，核对完整日志无ASan/UBSan告警。用户随后报告普通终端默认san的137项全部通过（4.33秒），代理未另行复核该次完整日志。覆盖手算方向、输出不足容量、严格平局、全部已选点判断、非有限距离、负内积、待追加点与小M线图导航；未做真实OOM/并发验证。
- SIFT前100000条/128维/10000查询、L2、k10、M16/Mmax0=32、efC200/efS64、seed42同轮比较完成；严格参考由截断库Flat重算。原始目录data/p2-results/heuristic-compare-ol50x8nm保存旧/新静态链接二进制、摘要、命令、源码和环境快照。环境i7-14650HX16核/24CPU、Ubuntu22.04/GCC11.4/RelWithDebInfo，单线程CPU0、powersave、允许Turbo、接电，warmup1/runs3；实际频率/温度/电源模式未确认。完整数值与局限见bench/hnsw.md；不与历史跨轮耗时直接相除，不宣称百万库或P2最终验收达成。
- 取舍ADR-0011，手算与面试表docs/learn/hnsw-neighbor-selection.md。剪枝块已提交，后续扫描及Faiss同轮对照已完成，见后续记录；局部查询分配成本仍保留，未经批准不新增公开工作区接口。

### 2026-10-23 完整SIFT1M与efSearch曲线（已提交）

- 已扩展bench/bench_main.cpp，--ef-search-list仅适用HNSW且与--ef-search互斥；严格正整数列表、保留顺序/重复，只建图一次，各ef独立warmup/计时/召回，--csv逐行刷新并记录请求/实际max(ef,k)值。新增stderr阶段信息，单值入口仍保留。未改核心索引接口、算法或局部查询工作区。
- tests/bench_cli_test.cmake新增1项CTest集成检查，无额外Python测试依赖；GCC11.4 rel/san各138项通过（137个GoogleTest+1个CLI），本轮san使用detect_leaks=0，完整日志无ASan/UBSan告警。用户随后报告普通终端默认san的138项全部通过（4.52秒），代理未另行复核该次完整日志。
- 全量1000000×128、10000查询、L2、k10、M16/Mmax0=32、efC200、seed42，efS16/32/64/128/256同图扫描已完成；官方groundtruth严格ID比较，不截断、不改平局口径。原始目录data/p2-results/sift1m-ef-sweep-1zq_3h4q，基于2afb07f及本轮bench修改，源码/二进制摘要、命令、逐阶段环境均保存。
- 环境i7-14650HX（16核/24逻辑CPU）、Ubuntu22.04/GCC11.4/RelWithDebInfo/AVX2Four，单线程CPU0、powersave、允许Turbo、接电，每ef预热1/正式3轮，频率/温度/电源模式未确认。曲线与指标见bench/hnsw.md；默认ef64的Recall/P99、构建耗时和索引容量达到相应目标，该扫描阶段当时尚未验证Faiss相对吞吐，后续结果见下节。内存统计区分索引申请容量与进程峰值RSS。
- 此前保存的汇总bench/results/p2-sift1m-ef-search-20261023.csv及SVG后续按用户决定仅在本地保留，绘图脚本bench/plot_hnsw_sweep.py仅可选依赖matplotlib；C++构建不依赖它。阶段范围docs/validation/p2.md，学习/面试表docs/learn/hnsw-ef-search-sweep.md。
- 当前本轮改动未提交，Git写操作仍由用户执行。下一步用户提交评测入口/结果，再准备同条件Python Faiss对照；不静默新增公开工作区接口，不因HNSW达标抹去P0原验收未达成记录。

### 2026-10-23 Python Faiss对照与收尾（当前交付）

- 用户已提交扫描入口/绘图/全量曲线，当前HEAD57a2591。用户选择venv，已在普通终端安装faiss-cpu1.12.0、numpy2.2.6、packaging25.0；解释器build-deps/faiss-venv/bin/python，实际后端AVX2。此前代理沙盒下载失败并发起过外部权限请求，用户中止后要求先商量；此后不再尝试下载或越过沙盒，依赖由用户完成。
- venv的include-system-site-packages=false，但PYTHONPATH含ROS Humble导致普通pip check看到launch-ros缺PyYAML；已只读确认包实际在venv，并用-I验证无损坏依赖/ROS不可见。后续使用解释器明确路径与-I，不依赖用户交互终端或改全局环境。
- 新增bench/faiss_bench.py和faiss-requirements.txt，tests/faiss_bench_test.py的8项可选Python检查已通过。Faiss不进入C++链接或默认CTest依赖，现有138项Cpp/CLI记录继续适用。公开search逐条查询，预分配D/I，Python包装成本包含且不扣除；M16/Mmax0=32、efC200、seed42、OMP1，Faiss默认bounded queue/relative distance策略显式保留。相同seed不保证与自研相同图。
- 全量同轮对照已在data/p2-results/faiss-compare-23lk4yyv成功完成（完整1M×128、10000查询、严格官方ID，单线程CPU0、warmup1/runs3，依次自研与Faiss；两进程均退出0）。环境i7-14650HX16核/24CPU、Ubuntu22.04/GCC11.4/RelWithDebInfo，自研AVX2Four、Faiss AVX2；阶段快照powersave、允许Turbo、接电，实际频率/温度/电源模式与背景负载未完整采样。命令、源码/二进制/数据SHA256、环境及CSV保存；未做顺序随机化/置信区间。
- Faiss memory_bytes明确标记logical_payload_bytes，统计可见数组有效字节，不能直接与自研MemoryUsage申请容量计算优劣。协议bench/faiss.md，学习/面试表docs/learn/faiss-baseline.md；结果已完成；默认ef64自研/Faiss的严格Recall与QPS同时报告，不能把同ef吞吐比当作同准确率的纯C++内核比较。仓库汇总见bench/faiss.md，完整CSV/SVG仅在本地bench/results及原始目录保留。

- 本轮默认ef64自研严格Recall达到0.95、相对本协议Python Faiss吞吐达到60%，五个ef点吞吐比均超过60%；自研P99、构建时间与索引容量也达到既有目标。P2主要本机验收已通过，不据此宣称原生Faiss同准确率比值、查询零分配或已验证真实并发/OOM。
- 收尾安全构建及rel/san各138项通过，san临时detect_leaks=0，核对完整日志无ASan/UBSan告警；用户此前默认san138项4.52秒记录保留。本轮未改Cpp实现，无需因Python工具重复请求用户跑默认san；Python8项检查通过，绘图完成并验证拒绝批量结果混入。
- 本轮新增Faiss脚本/固定依赖清单/可选测试、对照CSV及SVG，扩展绘图--compare并忽略Python字节码缓存；学习与面试表docs/learn/faiss-baseline.md。当前改动未提交，Git写操作仍由用户执行；下一步用户提交本轮工具/记录、将feat/p2-hnsw合回main，再决定P3开发分支与实现计划。未经批准不新增公开工作区接口，不擦除P0原验收未达成记录。

### 结果文件管理（用户已确认）

- 前两个Faiss工具/绘图提交已由用户完成。第三次提交只包含文档汇总、环境、复现命令和忽略规则，不添加运行生成的CSV/SVG。
- .gitignore已忽略/bench/results/；data/p2-results/原本已忽略。此前入库的两份p2-sift1m-ef-search-20261023.csv/svg由用户执行git rm --cached取消跟踪，保留本地文件，不重写旧提交。文档已移除对本地生成文件的图片/下载链接，使用生成命令代替。
- 仓库继续保存作者测量汇总，但使用者需自行运行并记录自己的环境；这不改变P2验收结果或严格Recall口径。

## 工作方式约定

### 工作模式：代理写实现，用户负责「能讲出来」

> **2026-10 调整。** 此前约定是「核心实现由用户手写，代理不得代写」。
> 用户已明确：**时间不足以亲手写实现，需要快速推进** —— 实现交给代理，
> 用户保证能讲清关键设计。**不要再回到旧的「不得代写」模式。**

**代理每交付一块代码，必须同时给出三样东西：**

| # | 交付物 | 要求 |
|---|---|---|
| 1 | **它解决了什么问题** | 一两句话。**不要复述代码** |
| 2 | **取舍记录** | 为什么选这个方案 · 代价是什么 · 还有哪些替代方案 |
| 3 | **面试讲点** | 可能的追问 + 答案要点，**表格形式**最好 |

**用户的责任：读懂 + 能复述。** 讲不出来的部分要回头问，不能跳过。

**代理的自检（发出交付前问自己）**：

> 这份交付**除了能跑，还能让他讲出来吗？**
> 不能 → 补上第 3 项再发。

### 决策权仍在用户手上

代码归属放宽了，但**决策权没有放宽**。以下必须问，不要自己拍板：

- 接口签名变更、数据结构选择的大方向
- 性能目标与验收标准
- 与面试 / 学习相关的取舍（哪块值得多花时间讲透）
- 任何会让已写代码大范围返工的改动

### 代理的行为准则

- ✅ **写实现** —— 算法 · 数据结构 · 文件 IO · 内存管理 · 并发 · 性能优化
- ✅ **讲原理** —— 每块代码的「为什么」，不只是「是什么」
- ✅ **定位问题** —— 读代码与报错，指出错在哪一行、为什么
- ✅ **审阅** —— 用户改过的代码，指出隐患与可改进处
- 🟡 **不要静默替用户做设计决定** —— 影响接口或整体结构的，先问一句
- ❌ **不要交付「能跑但讲不清」的代码** —— 那等于没交付

### 交付格式（建议）

写完一块代码后，回复按这个顺序：

1. **做了什么**（一两句）
2. **关键设计点**（2~4 条，每条一句话）
3. **面试讲点**（表格：会被问什么 / 怎么答）
4. **下一步**（需要用户做什么）

### git

- **git 的写操作（`add` / `commit` / `push` / `merge` / `rebase` / `reset`）由用户执行**
- 代理只做只读查询（`status` / `log` / `diff` / `show` / `branch`）
- 提交信息可以由代理**起草**，用户审阅后自行执行

### 其他

- 用户已授权代理自主执行沙盒内安全的构建、测试与测量；不要为常规执行重复请求确认，不越过沙盒权限，不执行危险命令；避免重复启动用户正在运行的长评测

- 构建最多使用 8 个并行任务（`-j 8` 或更小）；禁止不带数量的 `-j`，避免内存耗尽
- 破坏性操作（删除、批量重写）先 dry-run 再执行
- 涉及性能数字时，**必须同时记录测量环境**：CPU 型号 · 核心数 · 频率调节状态 · 是否 warmup · 单/多线程
- 注释用**中文**；字符串字面量、控制台输出、构建脚本的输出用**英文**


## P3 当前接续点（2026-10-23）

- 用户已完成 P2 收尾提交并合回 main（045f26b），已创建并切换到 feat/p3-ivf-and-quantization；Git 写操作仍由用户执行。上文 P2 历史记录中的“待提交/合并”已由此更新，不重做 P2 长评测。
- 当前实现 KMeans：公开签名沿用 docs/design/interfaces.md，L2 k-means++ 初始化、固定次数 Lloyd、空簇最远样本重置。距离及均值 double 累加，输出 float，等距选较小簇 ID；最终 ClusterSizes 对应交付质心的重新分配，重复点允许最终空簇。
- Train 成功才提交输出、统计及 RNG；同实例连续训练延续 RNG，同环境同种子调用序列复现，不保证跨标准库/libm 逐位一致。Assign 为 const，无成员写入；Train 不与同实例其他操作并发。输入需有限且 n>=k；不新增 metric 参数，不把 L2 聚类描述为 IP/余弦专用训练。
- 第一块仅聚类训练器，尚无 IVF/SQ/PQ、SIMD 训练优化、真实大数据训练或 P3 性能验收；复杂度 O(iters×n×k×dim)，空间 O(n+k×dim)。取舍 ADR-0012，学习与面试表 docs/learn/kmeans.md，验证记录 docs/validation/p3.md。
- 代理完成 GCC11.4 rel/san 各150项CTest（新增12项KMeans）；san临时detect_leaks=0，完整日志无ASan/UBSan告警，未验证泄漏。clang-format、diff检查通过；clang-tidy补齐GCC标准库路径后退出0，仍有风格建议。本块尚未提交，用户此前默认san138项不替代本块新增测试。


### P3 未量化 IVF（2026-10-23）

- 用户已提交 KMeans 实现63a9a52/文档49f661f，并创建 feat/p3-ivf-flat。用户要求较大且需要多次提交的功能使用独立分支，功能完成后合回 P3 集成分支。Git 写操作仍由用户执行。
- 用户已明确确认首版仅 L2，内积路由后续讨论；沿用既有静态设计：Add 累积、Build 成功后拒绝 Add，重复 Build 无操作；n<nlist 时 Build 报错，可追加后重试。Build 先局部训练/分配/生成 CSR，成功再 swap 提交，未注入真实 OOM。
- 新增 include/ann/ivf.h、src/ivf.cpp、私有 ivf_lists.h/.cpp。KMeans 全量训练、CSR 原始 ID 列表，距离候选使用既有 float 运行时派发；Search const 无成员写入、局部缓冲有分配，nprobe 原子请求值、0->1、查询时截断到 nlist。全探测跳过中心排序扫描全库；有限 float 距离溢出则报错，输出成功后才写。结果距离/ID排序、缺项填哨兵，不自动扩大 nprobe，不改变 P0 严格 Recall。
- 沿用总设计的 unique_ptr<Quantizer> 构造签名，新增设计已有的纯抽象 quantizer.h；当前非空量化器报错，尚无 SQ/PQ/codes/residuals。MemoryUsage 统计对象及成员数组申请容量，排除局部工作区，不是 RSS。
- 本轮先交付核心构建/查询，ann_bench 尚未支持 IVF，没有真实数据训练/性能/量化验收结论。取舍 ADR-0013，学习/面试表 docs/learn/ivf-flat.md；后续先提交核心与记录，再接入评测入口和 nprobe 对比。

- IVF核心代理完成rel/san各171项CTest（新增CSR5+IVF16），san临时detect_leaks=0，完整日志无ASan/UBSan告警；clang-format/diff检查通过，clang-tidy补齐GCC标准库路径后退出0但保留风格建议。尚未运行普通终端默认san171项。当前实现/记录未提交，下一块评测入口仍在同一功能分支推进。
