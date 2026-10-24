# 未量化 L2 IVF 评测

当前支持 IVFFlat，不支持 SQ/PQ 或内积。构建调用已有 KMeans 在全部添加数据上训练，尚无采样、SIMD 训练或训练并行；先用小规模确认行为，再扩大数据量。此处没有 P3 最终量化验收结论。

## 参数与统计口径

| 参数 | 含义 |
|---|---|
| --index ivf | 选择 IVFFlat；默认仍为 Flat |
| --nlist | 粗簇数，默认1024，必须不超过实际加载的base条数 |
| --kmeans-iters | 训练迭代次数，默认25 |
| --nprobe | 请求探测的列表数，默认16 |
| --nprobe-list | 正整数逗号列表，如1,4,16；与--nprobe互斥 |
| --seed | 同时用于合成数据和索引训练，默认42 |
| --csv | 保存每个参数点的指标，逐行刷新 |

IVF 专用参数只允许用于 --index ivf，HNSW 参数不能用于 IVF。新增正整数参数严格拒绝负数、零、空值、尾随字符和溢出。扫描列表保留输入顺序和重复项，支持反向/重复测量。

一次扫描只训练和建立一份索引，每个 nprobe 独立预热、计时和额外查询一轮统计召回。build_seconds 包含 Add 的复制以及 Build 的训练/分配/CSR构建；不含数据文件加载或参考答案计算。各行共享构建耗时及内存。有效探测数为 min(请求值,nlist)，候选不足不自动扩大范围。查询局部分配和逐查询时钟采样成本计入时间。

合成数据和截断库使用独立 Flat 重算严格 ID 答案；完整 SIFT 使用官方 groundtruth。不得把 IVF 自身输出当作参考。nprobe=nlist 消除粗路由遗漏，但平局选择仍可能与官方答案不同；P0 的严格 Recall 决定和原未达1.0记录保留。

CSV 保留原有23列的名称和顺序，末尾追加 nlist,nprobe,effective_nprobe,kmeans_iters，合计27列。Flat/HNSW 的新增四列为0，IVF 的HNSW参数列为0。已有按列名读取的 HNSW 绘图工具可继续使用；自行编写的固定列数读取器需要适配27列。不修改历史CSV。

## 构建与复现

在仓库根目录、既有依赖已配置时：

```bash
CCACHE_DISABLE=1 cmake --build --preset rel -j 8
ctest --preset rel
mkdir -p data/p3-results
ivf_result_dir=$(mktemp -d "$PWD/data/p3-results/ivf-XXXXXX")
date -Iseconds > "$ivf_result_dir/date.txt"
lscpu > "$ivf_result_dir/cpu.txt"
uname -a > "$ivf_result_dir/system.txt"
c++ --version > "$ivf_result_dir/compiler.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$ivf_result_dir/governor.txt"
cat /sys/devices/system/cpu/intel_pstate/no_turbo > "$ivf_result_dir/no-turbo.txt"

taskset -c 0 ./build-rel/bin/ann_bench \
  --index ivf --n 5000 --nq 100 --dim 16 --k 10 \
  --nlist 16 --kmeans-iters 5 --nprobe-list 1,4,16 \
  --metric l2 --seed 42 --warmup 1 --runs 3 \
  --csv "$ivf_result_dir/ivf.csv" \
  > "$ivf_result_dir/ivf-console.txt" 2> "$ivf_result_dir/ivf-progress.txt"
```

CPU0需在本机允许列表内；记录是否接电、电源模式，测前和测后保存频率调节状态。不要为了测量静默修改系统电源设置。需要同轮Flat对照时使用相同 n/nq/dim/k/seed/warmup/runs 和CPU绑定，只省略IVF专用参数。进程按顺序执行，避免同时跑造成干扰。

真实数据子集可使用 --data data --max-base 10000 --nlist 32 --kmeans-iters 5 --nprobe-list 1,4,16,32 替换上述合成数据参数。--data 的查询数取自文件，--nq 不截断磁盘查询；截断base时会重算该子集的Flat参考。该例尚未由本轮执行。当前训练复杂度 O(iters×n×nlist×dim)，不要把小规模训练耗时直接外推到全量百万数据。

## 本机小规模链路检查（2026-10-23）

基于7450fa6及本轮未提交评测入口，原始目录 data/p3-results/ivf-flat-synthetic-0vx5kj1o。包含命令、各进程前后环境、CPU/编译器/系统、源码差异及源码/二进制SHA256、控制台/阶段输出和CSV；两进程均退出0。CSV/日志只保留本地，不入库。

环境：i7-14650HX，16物理核/24逻辑CPU，Ubuntu22.04、GCC11.4、RelWithDebInfo、AVX2Four；单线程固定CPU0，powersave、允许Turbo，阶段快照接电。电源模式未确认，实际频率/温度未采样。合成5000×16、100查询、L2、k10、seed42，每点预热1轮/正式3轮，先Flat后IVF；IVF nlist16、iters5，仅训练一次。负载很短，没有随机化顺序/跨轮重复，不据此计算稳定加速比或外推真实数据性能。

| 索引 | 请求/有效nprobe | 严格Recall@10 | QPS（短负载观测） | P99 ms |
|---|---|---:|---:|---:|
| Flat | — | 1.000 | 41630.4 | 0.034180 |
| IVFFlat | 1/1 | 0.430 | 316628.4 | 0.003737 |
| IVFFlat | 4/4 | 0.816 | 136281.7 | 0.008437 |
| IVFFlat | 16/16 | 1.000 | 41898.4 | 0.030113 |

IVF 本轮构建0.006338s，MemoryUsage=341268字节；Flat构建0.000274s，MemoryUsage=320072字节。内存为对象和成员数组申请容量，不是峰值RSS，不含构建/查询工作区。这里的用途是确认独立参考、一次构建及探测范围，不代表量化压缩率或P3验收达标。

## 验证与后续

本轮 rel/san各172项CTest通过（170项GoogleTest+2项CLI）；san临时detect_leaks=0，完整日志无ASan/UBSan告警。新增CLI检查覆盖只构建一次、CSV参数、独立参考、重复点、全探测/超范围截断、单值与扫描、格式/互斥/错误索引和默认Flat兼容。原HNSW扫描检查已适配新增CSV列并通过。

下一步完成真实子集的nprobe测量，单独评估训练时间和查询召回，再决定是否需要训练采样/加速；这些改变不在本次实现中。随后接入SQ/PQ。学习/面试笔记在本机 docs/learn/ivf-benchmark.md，阶段验证见 docs/validation/p3.md。
