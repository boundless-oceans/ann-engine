# ann-engine

> **ANN** = *Approximate Nearest Neighbor*（近似最近邻）。
> 给定一个查询向量，在百万级向量库中找出**最接近**的那几个 ——
> **不保证精确，但快几个数量级**。这是向量检索、RAG、推荐系统的底层基础设施。

从零实现的高性能 ANN 检索引擎（C++20）。**手写** HNSW / IVF 索引与 PQ/SQ 量化，
不依赖 Faiss 等现成库，并与 Faiss 做性能对照。

> **状态：设计阶段** —— 设计文档已定稿，代码尚未开始。

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

> ℹ️ 设计文档正在本地持续演进，将在实现基本定型后一并合入本仓库。
> **当前点击下面的链接会 404，属预期。**

| 文档 | 内容 |
|---|---|
| [总设计](docs/design/design.md) | 三合一：**总设计**（目标 · 技术栈 · 架构 · 实现顺序 P0–P7 · 问题与对策）+ **架构图集 14 张** + **内存与 SIMD** |
| [类与接口](docs/design/interfaces.md) | 二合一：**UML 类图 + 分层细化 + 速查表** + **详细签名与成员布局** |
| [分布式扩展](docs/design/raft.md) | P7：Raft 共识层 / 状态机 / 传输 / 与索引的集成 |
| [技术决策记录](docs/decisions/) | ADR：每次关键取舍留一份（背景 / 候选 / 决定 / **代价**） |

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 工程骨架 + FlatIndex + 评测框架 | ⬜ 未开始 |
| P1 | SIMD 距离 + 内存池 | ⬜ 未开始 |
| P2 | HNSW | ⬜ 未开始 |
| P3 | IVF + SQ/PQ | ⬜ 未开始 |
| P4 | 持久化 + 崩溃恢复 | ⬜ 未开始 |
| P5 | 服务化（gRPC / HTTP） | ⬜ 未开始 |
| P6 | 差异化（过滤检索 / GPU 批量距离 / 删除） | ⬜ 未开始 |
| P7 | Raft 分布式扩展（多副本一致） | ⬜ 未开始 |

## 目录结构

> 下面是**规划中**的结构。目录会随实现进度逐步建立。

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
- GCC 12+（C++20；注意 GCC 11 缺 `<format>`）
- CMake 3.22+ · Ninja · ccache
- GoogleTest · Google Benchmark
- 构建步骤见 [总设计文档](docs/design/design.md)

## 数据集

SIFT1M（1M × 128d）。**不入库**（体积大），获取方式见 [data/README.md](data/README.md)。

## 命名与约定

- 命名空间 `ann`
- 索引统一实现 `Index` 抽象接口（见总设计文档 §2.2）
- **索引内部一律用下标（`uint32_t`）而非指针** —— 既利于 cache，也让持久化几乎免费
