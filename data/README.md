# 数据集

本目录**不入库**（SIFT1M 约 512 MB，且随时可重新下载）。
本文件是唯一被仓库跟踪的文件。

## 需要哪些文件

| 文件 | 规模 | 说明 |
|---|---|---|
| `sift_base.fvecs` | 1,000,000 × 128d | 库向量（≈ 512 MB） |
| `sift_query.fvecs` | 10,000 × 128d | 查询向量 |
| `sift_groundtruth.ivecs` | 10,000 × 100 | **标准答案**：每个 query 的 100 个真实最近邻（暴力检索所得） |

> `sift_groundtruth.ivecs` 是算 Recall@K 的基准。**没有它就无法验证近似索引的正确性。**

## 获取方式

### 方式一：corpus-texmex（经典 fvecs 格式，**推荐**）

来源：IRISA 的 ANN 数据集站（corpus-texmex），包名 `sift.tar.gz`。
官网可能较慢，可用镜像站。解压后把三个文件放到本目录。

**推荐理由**：格式极简，自己写读取器只要 20 行（见下），不需要引入任何额外依赖。

### 方式二：ann-benchmarks（HDF5）

`ann-benchmarks` 的 `sift-128-euclidean.hdf5` 一个文件含 base / query / groundtruth，
并附带标准评测脚本。

**代价**：C++ 读 HDF5 要引入 HDF5 依赖，比写 fvecs 读取器麻烦。**建议先用方式一。**

## fvecs / bvecs / ivecs 格式（自己写读取器用）

三种格式结构相同，只是元素类型不同：**「维度 + 数据」反复出现**。

```
fvecs:  [int32 dim][float32 × dim]   ← base / query
bvecs:  [int32 dim][uint8   × dim]   ← 二值/量化向量
ivecs:  [int32 dim][int32   × dim]   ← groundtruth
```

```
以 fvecs 为例（dim = 128）：

  偏移 0    ┌──────────────┐
            │ int32 dim=128│
  偏移 4    ├──────────────┤
            │ float × 128  │  ← 第 1 个向量（512 B）
  偏移 516  ├──────────────┤
            │ int32 dim=128│  ← 第 2 个向量的维度
  ...       └──────────────┘
```

**实现要点**

- 全部为**小端**（x86 / ARM 均为小端，直接 `fread` 即可，无需转换）
- 文件大小 / `(4 + 4×dim)` = 向量个数，可用来自检
- **必须检查 `fread` 的返回值** —— 不要假设一次读完
- 可以把整个文件一次性读进内存，再按下标切片（比逐条读快得多）

## 小规模调试集

调正确性时**不要用 100 万条**。建议先切一个子集：

```bash
# 只取前 10 万条做调试（自己写个脚本或用 head 按字节截断）
head -c $((100000 * 516)) sift_base.fvecs > sift_base_100k.fvecs
```

> ⚠️ 截断后要**重新计算 ground truth**（用你的 FlatIndex 对子集重跑），
> 不能沿用完整的 `sift_groundtruth.ivecs`。
