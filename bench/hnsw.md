# HNSW 基础索引评测

当前已接入多样性启发式选邻与反向超限裁剪，下面保留此前最近 M 基线及同轮对照。`Search` 每次创建局部工作区，包含内存分配及 O(N) 访问表初始化成本；本页不代表 P2 最终验收。

## 运行

```bash
CCACHE_DISABLE=1 cmake --build --preset rel -j 8
taskset -c 0 ./build-rel/bin/ann_bench \
  --index hnsw --n 5000 --nq 100 --dim 16 --k 10 \
  --m 16 --ef-construction 200 --ef-search 64 --seed 42 \
  --warmup 1 --runs 3
```

默认仍是 `--index flat`。`--m` 同时设置 M 和 Mmax0=2M；`--seed` 同时用于合成数据和随机层数。构建搜索实际使用 max(ef_construction,M)，查询实际使用 max(ef_search,k)。

合成数据的 HNSW 答案由独立 FlatIndex 计算，严格比较 top-k ID；生成参考答案不计入查询计时。不能拿 HNSW 自己的结果作答案。Flat 的合成数据路径仍是自算答案，只检查评测链路。

完整 SIFT1M 可使用下面的命令，但本轮未运行，建图可能较久：

```bash
./build-rel/bin/ann_bench --index hnsw --data data \
  --k 10 --metric l2 --m 16 --ef-construction 200 --ef-search 64 \
  --warmup 1 --runs 3
```

SIFT 官方答案仅用于 L2；程序拒绝 `--data` 配合 `--metric ip`。截断 base 时不能直接把完整库官方 ID 当作截断库答案，应遵循评测程序已有的参考答案重算路径。

## 2026-10-23 小规模基线

原始记录：`data/p2-results/nearest-m-final-us017dob/`，包含 console.txt、context.json、cpu.txt。构建基于提交 `93a00c6` 加本轮未提交修改，源码 SHA256 已保存在 context.json。

环境：Intel Core i7-14650HX，16 物理核/24 逻辑 CPU，Ubuntu 22.04、GCC 11.4、RelWithDebInfo；单线程绑定 CPU 0（P 核），CPU0 governor=powersave，intel_pstate/no_turbo=0（允许 Turbo），实际频率未采样。ACAD/online=0，未确认接电；platform_profile 不可读，未确认电源模式，不沿用前一天的接电/平衡状态。预热 1 轮，正式 3 轮。

数据：合成 5000×16，查询 100，L2，k=10，seed=42；M=16、Mmax0=32、ef_construction=200、ef_search=64，后端 AVX2Four。

| 指标 | 结果 |
|---|---:|
| 严格 Recall@10（独立 Flat 参考） | 0.999000 |
| 构建时间 | 0.674 s |
| 索引驻留申请容量 | 1.6 MiB |
| QPS | 20610.8 |
| mean / P50 | 0.0485 / 0.0481 ms |
| P95 / P99 / max | 0.0594 / 0.0620 / 0.0718 ms |
| 最高层 / L0 平均出度 | 4 / 24.96 |

内存数值不包含构建临时工作区、查询临时工作区，也不是进程峰值 RSS；输出中的 MB 实际按 MiB 换算。局部查询工作区创建位于 Search 内，已计入查询耗时。这里没有同轮 Flat 性能对照，不能据此给出端到端加速比，也不能外推 SIFT1M Recall。

## 2026-10-23 SIFT 前 10 万条基线

原始目录：`data/p2-results/nearest-m-sift100k-a1nvea4c/`。基于干净提交 `52abb07`，源码摘要、命令、CPU 和电源快照随结果保存。

环境：Intel Core i7-14650HX，16 物理核/24 逻辑 CPU，Ubuntu 22.04、GCC11.4、RelWithDebInfo（ASan/UBSan OFF）；两种索引顺序执行，单线程绑定 CPU0（P 核），powersave，允许 Turbo，实际频率未采样。开始及完成后快照 ACAD/online=1（接电），platform_profile 不可读，电源模式未确认。均预热1轮、正式3轮。

使用原 SIFT1M 的前100000条、128维、全部10000条查询，L2、k=10。因为 base 截断，清除完整库的官方 groundtruth，由 Flat 在截断库上重新计算严格 ID 参考。它是 SIFT 子集基线，不是百万库官方验收。HNSW 的 M=16、Mmax0=32、ef_construction=200、ef_search=64、默认seed=42，后端 AVX2Four。

```bash
taskset -c 0 ./build-rel/bin/ann_bench \
  --index hnsw --data data --max-base 100000 --k 10 --metric l2 \
  --m 16 --ef-construction 200 --ef-search 64 --warmup 1 --runs 3

taskset -c 0 ./build-rel/bin/ann_bench \
  --index flat --data data --max-base 100000 --k 10 --metric l2 \
  --warmup 1 --runs 3
```

| 指标 | HNSW 最近 M | Flat |
|---|---:|---:|
| 严格 Recall@10 | 0.968190 | 1.000000（自算精确参考） |
| 构建时间（s） | 23.439 | 0.033 |
| 索引驻留申请容量（MiB） | 81.1 | 48.8 |
| QPS | 8916.6 | 306.4 |
| mean / P50（ms） | 0.1121 / 0.1119 | 3.2635 / 3.2546 |
| P95 / P99 / max（ms） | 0.1497 / 0.1751 / 0.2705 | 3.7563 / 4.0674 / 4.9684 |

HNSW 最高层5、L0平均出度24.98。本轮吞吐约为 Flat 的29.1倍，但以召回损失、较长构建时间和较大索引容量为代价；不能当作同准确率加速比或外推百万库收益。3轮合计30000个正式延迟样本，Recall 另用10000条查询计算。两种索引顺序运行，没有控制实际频率/温度，也没有多次独立建图重复实验；后续修改算法仍需同轮重测旧版对照，不能将本轮耗时直接与另一天测量相除。

每种索引都会单独计算参考答案，并额外执行一轮查询统计 Recall；这些不计入正式查询时间，因此终端总耗时会超过“建图时间+3轮查询”。HNSW 查询计时包含局部工作区分配和访问表初始化。内存仍不含临时空间，不是 RSS；加载器先读取完整 base 再截断，进程峰值内存也会受完整数据加载影响。

## 2026-10-23 多样性剪枝同轮对照

原始目录：`data/p2-results/heuristic-compare-ol50x8nm/`。变更前保存最近 M 的静态链接可执行文件，再构建剪枝版本；目录包含两份可执行文件及其 SHA256、源码快照/摘要、命令、CPU快照、逐版本开始/结束环境和原始输出。最近 M 对应提交2245169的代码，剪枝版本为其上的本轮未提交修改。未新增公开策略开关。

环境：Intel Core i7-14650HX（16物理核/24逻辑CPU）、Ubuntu22.04、GCC11.4、RelWithDebInfo、AVX2Four、单线程固定CPU0（P核）。两个版本顺序执行，均预热1轮、正式3轮；各版本前后CPU0 governor=powersave、no_turbo=0（允许Turbo）、ACAD/online=1（接电），电源模式未确认。实际频率、温度未采样，未做多次独立建图或交换执行顺序的实验，因此耗时差是本轮观测，不能单独归因于算法。

数据和参数完全相同：SIFT前100000条×128维、10000查询、L2、k10、M16/Mmax0=32、efC200/efS64、seed42。截断后由独立Flat重新计算参考，严格ID Recall不变。运行命令与前节相同，分别将可执行文件替换为目录中的 ann_bench_nearest / ann_bench_heuristic，并显式指定 --seed 42；完整命令已保存到 context.json。

| 指标 | 最近 M（本轮重测） | 多样性剪枝 |
|---|---:|---:|
| 严格 Recall@10 | 0.968190 | 0.988820 |
| QPS | 15315.9 | 14219.7 |
| 构建时间（s） | 13.484 | 15.386 |
| 索引驻留申请容量（MiB） | 81.1 | 81.1 |
| mean / P50（ms） | 0.0653 / 0.0657 | 0.0703 / 0.0703 |
| P95 / P99 / max（ms） | 0.0844 / 0.0996 / 0.1694 | 0.0893 / 0.1088 / 1.1438 |
| 最高层 / L0平均出度 | 5 / 24.98 | 5 / 19.75 |

严格Recall增加2.063个百分点；同轮观测QPS约低7.2%，构建约慢14.1%。更少但分布更广的连接有助于本数据上的召回，但查询和建图并未同时加速。出度减少不等于申请容量减少：图仍用定长槽位，MemoryUsage也排除临时工作区。它不是同准确率性能对照；需要后续efSearch曲线比较。结果不代替全量SIFT1M验收或Faiss基线。

历史最近M的QPS为8916.6，本轮重测为15315.9，说明跨轮耗时不能直接相除；本节只用同轮版本进行比较，不把历史环境差异猜测成确定原因。

## 验证与待办

新增 13 个索引测试，rel/san 各 127 个通过；san 在沙盒内设置 `ASAN_OPTIONS=detect_leaks=0`，无 ASan/UBSan 告警。用户随后报告普通终端默认 san 的 127 项全部通过，用时 5.78 秒；代理未另行复核该次完整日志。复测命令：

```bash
ctest --preset san --output-on-failure
```

最近 M 基础阶段的后续剪枝及同轮对照现已完成，见上节；尚未进行全量 SIFT1M、Faiss 对照、真实并发/TSan 或分配失败注入验证。

当前剪枝块新增10个测试，总计rel/san各137项通过；沙盒san设置detect_leaks=0，完整日志无ASan/UBSan告警。用户随后报告普通终端默认san的137项全部通过（4.33秒），代理未另行复核该次完整日志。手算、边界和面试点见本地 [多样性选邻笔记](../docs/learn/hnsw-neighbor-selection.md)。下一步提交当前块，再进行efSearch调参与全量数据验证；公开查询仍使用局部工作区。
