# ann-engine

> **ANN** = *Approximate Nearest Neighbor*（近似最近邻）。
> 给定一个查询向量，在百万级向量库中找出**最接近**的那几个 ——
> **不保证精确，但快几个数量级**。这是向量检索、RAG、推荐系统的底层基础设施。

从零实现的高性能 ANN 检索引擎（C++20）。**手写** HNSW / IVF 索引与 PQ/SQ 量化，
不依赖 Faiss 等现成库，并与 Faiss 做性能对照。

> **状态：P0 验证中** —— Ubuntu 上 rel 构建的 27 个测试已通过，合成数据评测已跑通；
> 全量 SIFT1M 已测得 Recall@10=0.999300，尚未达到 1.0；待排查差异并确认 Sanitizer 测试汇总。

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
| P0 | 工程骨架 + FlatIndex + 评测框架 | 🟡 rel 测试通过；全量 SIFT1M Recall=0.999300，差异待排查 |
| P1 | SIMD 距离 + 内存池 | ⬜ 未开始 |
| P2 | HNSW | ⬜ 未开始 |
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

待排查：逐条对照官方 top-10，定位缺失与额外返回的 ID，核对它们的实际距离。
相同距离下的边界选择、浮点计算及检索逻辑均是待核查项，目前没有确认原因。
保持 `Recall@10 == 1.0` 的原验收标准，不把本次接近 1.0 的结果记为验收通过。
Sanitizer 已由用户反馈执行完成，但尚未提供最终测试汇总及无告警确认。

## 数据集

SIFT1M（1M × 128d）。**不入库**（体积大），获取方式见 [data/README.md](data/README.md)。

## 命名与约定

- 命名空间 `ann`
- 索引统一实现 `Index` 抽象接口（见总设计文档 §2.2）
- **索引内部一律用下标（`uint32_t`）而非指针** —— 既利于 cache，也让持久化几乎免费
