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

真实数据子集可使用 --data data --max-base 10000 --nlist 32 --kmeans-iters 5 --nprobe-list 1,4,16,32 替换上述合成数据参数。--data 的查询数取自文件，--nq 不截断磁盘查询；截断base时会重算该子集的Flat参考。该配置现已完成，见后面的真实子集记录。当前训练复杂度 O(iters×n×nlist×dim)，不要把小规模训练耗时直接外推到全量百万数据。

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

## SIFT 前10,000条真实子集（2026-10-24）

基于干净提交1999ff4，在feat/p3-ivf-flat顺序运行Flat和IVF。原始目录data/p3-results/ivf-flat-sift10k-t_pu4_gx，保存CMakeCache、构建日志、CPU/编译器/系统、数据与二进制SHA256、完整命令、进程前后环境、控制台/阶段输出和CSV；两个进程均退出0。构建检查为no work to do，核心代码未改。本轮只测量和补文档，未重复运行既有172项CTest。

数据为SIFT原始base前10,000条、128维，使用全部10,000条查询，L2、k10；因截断库，官方groundtruth弃用，在同一子集上由Flat重新计算严格ID参考。IVF nlist32、kmeans_iters5、seed42；nprobe1/4/16/32共用一份索引。训练参数是起步基线，不声称最优或已充分收敛。

环境：i7-14650HX，16物理核/24逻辑CPU，Ubuntu22.04、GCC11.4、RelWithDebInfo、AVX2Four；单线程固定CPU0。2026-10-24 20:13:08–20:13:33（UTC+8）内顺序测量，阶段快照powersave、允许Turbo、接电，每点预热1轮/正式3轮。电源模式未确认，实际频率/温度未采样；记录了边界loadavg，但未完整采样背景负载。未随机化顺序、重复训练或计算置信区间。

| 索引 | nprobe | 严格Recall@10 | QPS | P99 ms |
|---|---:|---:|---:|---:|
| Flat | — | 1.000000 | 8211.3 | 0.160049 |
| IVFFlat | 1 | 0.566530 | 130645.5 | 0.013435 |
| IVFFlat | 4 | 0.925550 | 41114.7 | 0.038262 |
| IVFFlat | 16 | 0.999910 | 11781.0 | 0.126732 |
| IVFFlat | 32 | 1.000000 | 5879.0 | 0.240454 |

IVF构建0.179328s，MemoryUsage=5176692字节；Flat构建0.000543s，MemoryUsage=5120072字节。内存均为对象与成员数组申请容量，不是RSS，也不包含临时空间；未量化IVF没有压缩原始向量，另有质心、偏移和ID数组。

本轮全探测召回达到1.0，但QPS低于Flat。列表/间接访问和不同top-k实现都有成本，没有perf定位，不将差异归因于某个已证明的瓶颈。减少探测可提高本轮QPS，同时可能漏簇；选择工作点必须同时报告召回和吞吐。这是10k子集基线，不是全量SIFT1M、Faiss IVF对照或P3压缩≥8倍/Recall≥0.90的最终验收。P0严格1.0原未达标记录保持不变。

复现本轮参数（先按前文保存环境，再顺序执行）：

```bash
mkdir -p data/p3-results
ivf_subset_dir=$(mktemp -d "$PWD/data/p3-results/ivf-sift10k-XXXXXX")
taskset -c 0 ./build-rel/bin/ann_bench \
  --index flat --data data --max-base 10000 --k 10 --metric l2 \
  --seed 42 --warmup 1 --runs 3 --csv "$ivf_subset_dir/flat.csv" \
  > "$ivf_subset_dir/flat-console.txt" 2> "$ivf_subset_dir/flat-progress.txt"
taskset -c 0 ./build-rel/bin/ann_bench \
  --index ivf --data data --max-base 10000 --k 10 --metric l2 \
  --nlist 32 --kmeans-iters 5 --nprobe-list 1,4,16,32 \
  --seed 42 --warmup 1 --runs 3 --csv "$ivf_subset_dir/ivf.csv" \
  > "$ivf_subset_dir/ivf-console.txt" 2> "$ivf_subset_dir/ivf-progress.txt"
```

## 验证与后续

本轮 rel/san各172项CTest通过（170项GoogleTest+2项CLI）；san临时detect_leaks=0，完整日志无ASan/UBSan告警。新增CLI检查覆盖只构建一次、CSV参数、独立参考、重复参数点、全探测/超范围截断、单值与扫描、格式/互斥/错误索引和默认Flat兼容。原HNSW扫描检查已适配新增CSV列并通过。

真实子集的nprobe起步测量已完成。下一步提交本轮记录后，可将IVFFlat功能分支合回P3集成分支，再准备SQ8功能分支；大规模训练采样/加速仍需后续证据和设计确认，不在本轮中静默加入。学习/面试笔记在本机 docs/learn/ivf-benchmark.md，阶段验证见 docs/validation/p3.md。
