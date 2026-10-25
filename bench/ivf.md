# L2 IVF 与残差 SQ8 评测

当前支持 IVFFlat 与残差 IVFSQ8，不支持 PQ 或内积。构建调用已有 KMeans 在全部添加数据上训练，尚无采样、SIMD 训练或训练并行；先用小规模确认行为，再扩大数据量。此处没有 P3 最终量化验收结论。

## 参数与统计口径

| 参数 | 含义 |
|---|---|
| --index ivf | 选择 IVF；默认无量化器，程序默认索引仍为 Flat |
| --quantizer | none（默认）或 sq8，仅适用 IVF；sq8 编码残差 |
| --nlist | 粗簇数，默认1024，必须不超过实际加载的base条数 |
| --kmeans-iters | 训练迭代次数，默认25 |
| --nprobe | 请求探测的列表数，默认16 |
| --nprobe-list | 正整数逗号列表，如1,4,16；与--nprobe互斥 |
| --seed | 同时用于合成数据和索引训练，默认42 |
| --csv | 保存每个参数点的指标，逐行刷新 |

IVF 专用参数只允许用于 --index ivf，HNSW 参数不能用于 IVF。新增正整数参数严格拒绝负数、零、空值、尾随字符和溢出。扫描列表保留输入顺序和重复项，支持反向/重复测量。

一次扫描只训练和建立一份索引，每个 nprobe 独立预热、计时和额外查询一轮统计召回。build_seconds 包含 Add 的复制以及 Build 的训练/分配/CSR构建；不含数据文件加载或参考答案计算。各行共享构建耗时及内存。有效探测数为 min(请求值,nlist)，候选不足不自动扩大范围。查询局部分配和逐查询时钟采样成本计入时间。

合成数据和截断库使用独立 Flat 重算严格 ID 答案；完整 SIFT 使用官方 groundtruth。不得把 IVF 自身输出当作参考。nprobe=nlist 消除粗路由遗漏，但平局选择仍可能与官方答案不同；P0 的严格 Recall 决定和原未达1.0记录保留。

CSV 保留既有27列的名称和顺序，末尾追加 quantizer,encoding,code_size,lut_floats，合计31列。此前的 nlist,nprobe,effective_nprobe,kmeans_iters 仍在原位置。非IVF的这些IVF数值列为0，IVF的HNSW参数列为0。量化元数据分别为none/raw/0/0或sq8/residual/dim/(256×dim)，code_size单位为字节，lut_floats是float个数。已有按列名读取的HNSW绘图仍可用，固定列数读取器需要适配31列。不修改历史CSV。

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

## 残差 SQ8 同轮起步对照（2026-10-25）

基于ff402cf及本轮评测入口改动，先IVFFlat再IVFSQ8，均为SIFT原始base前10000条×128维，仅使用原始查询前100条。nlist32、kmeans_iters5、seed42、L2/k10，nprobe1/4/16/32。每模式构建一次，各点预热1/正式3轮，另跑一轮召回；两进程退出0。查询工作区分配、逐列表LUT重建与ADC均计入SQ8耗时。SQ8粗路由后端为AVX2Four，候选为标量ADC，不能把它标成全程AVX2距离内核。

原始目录data/p3-results/sq8-sift10k-q100-ejxigkwi，保存context、CPU/编译器/系统、CMakeCache、源码差异与源码/二进制/数据SHA256、命令、阶段前后环境、完整输出/CSV。workload子目录通过符号链接读取原始完整base，只复制查询/参考文件的前100条；--max-base 10000触发加载器丢弃官方答案，双方都在当前子集上独立Flat重算严格ID参考。没有修改原始数据文件；前缀官方参考仅用于加载器条数匹配，不用于本次Recall。

环境：i7-14650HX，16物理核/24逻辑CPU，Ubuntu22.04、GCC11.4、RelWithDebInfo；单线程CPU0，阶段快照powersave、允许Turbo、接电，2026-10-25 20:39:35–20:39:41（UTC+8）。电源模式未确认，实际频率/温度及背景负载未完整采样，记录了边界loadavg；未随机化顺序、重复建模或统计置信区间。这是100查询的起步观测，不能与前一天10000查询的QPS直接相除。

| nprobe | IVFFlat严格Recall | IVFSQ8严格Recall | IVFFlat QPS | IVFSQ8 QPS | IVFFlat P99 ms | IVFSQ8 P99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.605000 | 0.603000 | 122528.7 | 7830.6 | 0.014461 | 0.182224 |
| 4 | 0.931000 | 0.923000 | 38111.0 | 2030.8 | 0.044007 | 0.582659 |
| 16 | 1.000000 | 0.989000 | 10560.9 | 527.7 | 0.125931 | 2.087934 |
| 32 | 1.000000 | 0.989000 | 5070.6 | 257.7 | 0.256832 | 3.979110 |

IVFFlat构建0.221783s、索引申请容量5176732字节；IVFSQ8构建0.189911s、1337828字节。容量比约3.87，编码载荷本身为4倍；总体还包括列表/中心/模型/对象。不是进程RSS或构建峰值，加载器仍保留数据集、构建阶段还有残差临时矩阵。本轮构建耗时单次观测不能说明SQ8训练更快。

SQ8本轮节省索引容量，但没有查询加速，严格召回略降。全探测消除了漏簇，仍保留量化误差，因此不要求SQ8 Recall为1。已知实现每个非空探测簇建256×dim表并逐候选标量ADC，128维LUT为32768个float（128KiB），同查询复用一块缓冲但仍重复计算表；尚无阶段计时/perf，不能把全部耗时差直接归因于建表或缓存瓶颈。未静默改算法来美化结果，也不据此宣称压缩带来加速。PQ≥8倍/Recall≥0.90的P3最终目标尚未验收。

复现相同查询前缀（先按前文保存测量环境，需原始完整SIFT文件）：

```bash
mkdir -p data/p3-results
sq8_result_dir=$(mktemp -d "$PWD/data/p3-results/sq8-compare-XXXXXX")
python3 - "$sq8_result_dir" <<'PYDATA'
from pathlib import Path
import struct
import sys
root = Path.cwd()
view = Path(sys.argv[1]) / "workload"
view.mkdir()
base = root / "data/sift_base.fvecs"
with base.open("rb") as source:
    dim = struct.unpack("<i", source.read(4))[0]
assert dim > 0 and base.stat().st_size // (4 * (dim + 1)) > 10000
(view / "sift_base.fvecs").symlink_to(base)
for name in ("sift_query.fvecs", "sift_groundtruth.ivecs"):
    with (root / "data" / name).open("rb") as source:
        width = struct.unpack("<i", source.read(4))[0]
        assert width > 0
        source.seek(0)
        payload = source.read(100 * 4 * (width + 1))
        assert len(payload) == 100 * 4 * (width + 1)
    (view / name).write_bytes(payload)
PYDATA
for mode in none sq8; do
  taskset -c 0 ./build-rel/bin/ann_bench \
    --index ivf --quantizer "$mode" --data "$sq8_result_dir/workload" \
    --max-base 10000 --nlist 32 --kmeans-iters 5 --nprobe-list 1,4,16,32 \
    --k 10 --metric l2 --seed 42 --warmup 1 --runs 3 \
    --csv "$sq8_result_dir/$mode.csv" \
    > "$sq8_result_dir/$mode-console.txt" 2> "$sq8_result_dir/$mode-progress.txt"
done
```

需完整查询时使用原data目录；--nq不截断--data查询，不应误以为加--nq 100就只测100条。没有必要为了复现起步记录立即启动全量10000查询或百万库长测。生成文件仍不入库。

## 验证与后续

此前IVFFlat入口阶段 rel/san各172项CTest通过（170项GoogleTest+2项CLI）；san临时detect_leaks=0，完整日志无ASan/UBSan告警。新增CLI检查覆盖只构建一次、CSV参数、独立参考、重复参数点、全探测/超范围截断、单值与扫描、格式/互斥/错误索引和默认Flat兼容。原HNSW扫描检查已适配新增CSV列并通过。

IVFFlat真实子集与SQ8起步对照已完成。下一步提交本轮入口/记录后，可将SQ8功能分支合回P3集成分支，再准备PQ；SQ8速度优化应先分解成本，不在本轮静默改训练或残差方案。学习/面试笔记在本机 docs/learn/ivf-benchmark.md，阶段验证见 docs/validation/p3.md。


2026-10-25的SQ8入口新增1项CLI检查，全量rel/san各202项通过（199个GoogleTest+3个CLI），san临时detect_leaks=0，完整日志无ASan/UBSan告警。三套CLI检查通过，覆盖SQ8扫描/独立参考/元数据/内存降低/重复与单值一致/非法参数，默认Flat及原IVF/HNSW保持可用。本轮不改核心算法，后续是否优化SQ8逐列表建表应先定位成本；当前作为正确性与容量基线保留，再推进PQ，不能声称SQ8速度验收已达成。学习与面试表在本机docs/learn/sq8-benchmark.md。
