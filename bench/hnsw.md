# HNSW 基础索引评测

当前是最近 M 邻居连接的基线，尚未接入多样性启发式剪枝。`Search` 每次创建局部工作区，包含内存分配及 O(N) 访问表初始化成本；本页不代表 P2 最终验收。

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

## 验证与待办

新增 13 个索引测试，rel/san 各 127 个通过；san 在沙盒内设置 `ASAN_OPTIONS=detect_leaks=0`，无 ASan/UBSan 告警。用户随后报告普通终端默认 san 的 127 项全部通过，用时 5.78 秒；代理未另行复核该次完整日志。复测命令：

```bash
ctest --preset san --output-on-failure
```

后续：记录较大数据集的最近 M 基线，再接多样性启发式剪枝并同轮对照；尚未进行全量 SIFT1M、Faiss 对照、真实并发/TSan 或分配失败注入验证。
