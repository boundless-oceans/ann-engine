# Python Faiss HNSW 对照

Faiss 只作为外部 Python 基线，不链接进 ann_core。脚本为 faiss_bench.py，固定依赖为 faiss-requirements.txt；不改变 C++ 索引、严格 Recall 或已确认的局部查询工作区方案。

## 固定环境

本轮使用 Python3.10.12、PyPI faiss-cpu1.12.0、NumPy2.2.6、packaging25.0；实际 Faiss 编译选项为 OPTIMIZE AVX2。PyPI wheel由faiss-wheels项目打包，不冒充与conda或自编译Faiss具有相同的全部构建选项。

用户已在普通终端完成安装。重新准备时可以执行：

```bash
python3 -m venv build-deps/faiss-venv
build-deps/faiss-venv/bin/python -I -m pip install --only-binary=:all: \
  --index-url https://pypi.org/simple -r bench/faiss-requirements.txt
build-deps/faiss-venv/bin/python -I -m pip check
```

`pyvenv.cfg` 设置 include-system-site-packages=false。主机 PYTHONPATH 含 ROS Humble 路径，会让普通pip check看见launch-ros缺PyYAML的提示；`-I` 忽略该路径，已验证隔离模式pip check无损坏依赖。不为此修改ROS或系统Python，不需要sudo，也不依赖当前终端是否activate。

## 命令

```bash
mkdir -p data/p2-results
taskset -c 0 build-deps/faiss-venv/bin/python -I bench/faiss_bench.py \
  --data data --k 10 --m 16 --ef-construction 200 --seed 42 \
  --ef-search-list 16,32,64,128,256 --warmup 1 --runs 3 \
  --csv data/p2-results/faiss.csv --metadata data/p2-results/faiss-metadata.json

build-deps/faiss-venv/bin/python -I tests/faiss_bench_test.py -v
```

脚本在本进程内设置 OpenMP/OpenBLAS/MKL/BLIS 单线程预算，并调用 faiss.omp_set_num_threads(1)。不修改系统电源模式或其他终端环境。测试仅在装有Faiss的环境中手动执行，不让默认C++构建/CTest依赖Faiss或NumPy。

## 对照口径

| 项目 | 做法 |
|---|---|
| 数据与参考 | 完整SIFT1M与全部10000查询，L2、k10、官方答案，严格比较ID；不使用R@1或距离等价替代指标。 |
| 查询粒度 | 每次只查1条，双方均逐条计时；不能把Faiss整批查询吞吐与自研单条查询直接相除。 |
| 输出缓冲 | Faiss通过公开search的D/I参数复用结果数组，查询切片提前准备；不在计时中反复分配返回数组。 |
| Python成本 | Python公开包装器、绑定调用与Faiss内部查询都包含在延迟中，吞吐还包含Python循环/样本记录；不估算扣除开销，不称为纯原生C++对照。 |
| 计时 | 每ef独立预热1轮、正式3轮；额外一轮逐条查询统计召回，不计入QPS。分位数采用与C++相同的最近秩法。 |
| 图参数 | M16、L0容量32、efConstruction200、seed42；每边只建一次图。相同参数不等于相同随机层数、图或召回，Faiss内部插入顺序及RNG均有差异。 |
| Faiss搜索策略 | 保留并记录bounded queue=true、check_relative_distance=true、keep_max_size_level0=false；不为了追平自研实现而修改其算法。 |
| 内存 | 自研MemoryUsage是申请容量；Faiss Python接口下统计向量、levels、offsets、neighbors、概率元数据等数组的有效元素字节。后者不含剩余容量和对象开销，CSV标为logical_payload_bytes；两者不能直接当成同口径内存优势比较。 |

脚本支持 --synthetic 做链路检查，使用独立Faiss Flat参考；--max-base截断真实库后重算该子集的精确参考。合成生成器与C++不同，不能仅按相同seed把两份独立合成数据当成相同输入。正式对照使用相同的磁盘SIFT文件，并保存文件摘要。

同轮顺序测量仍存在温度/频率漂移和执行顺序影响，不声称随机化或置信区间。需要同时报告同ef的严格Recall与QPS比值；同ef不同召回不等于完全相同准确率的比较。

## 验证

8项Python检查已通过：fvecs/ivecs及坏文件、严格Recall/哨兵/重复ID、最近秩分位数、ef列表、固定种子建图、只读查询以及CLI重复ef/CSV/元数据。它们不是并发、OOM、Faiss源码构建或新的C++Sanitizer测试。

## 参考资料

- [Faiss v1.12.0 HNSW参数与默认邻居容量](https://github.com/facebookresearch/faiss/blob/v1.12.0/faiss/impl/HNSW.h)
- [Faiss v1.12.0 Python search及预分配D/I接口](https://github.com/facebookresearch/faiss/blob/v1.12.0/faiss/python/class_wrappers.py)
- [官方百万向量评测：线程数和查询批量会改变吞吐](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)
- [固定PyPI wheel及打包说明](https://pypi.org/project/faiss-cpu/1.12.0/)


## 2026-10-23 完整 SIFT1M 同轮结果

原始目录：`data/p2-results/faiss-compare-23lk4yyv/`，基于干净提交57a2591及本轮Python工具；包含数据/源码/二进制SHA256、完整命令、软件版本、逐阶段环境、输出与集成日志。先自研后Faiss，未并行测量，运行区间18:09:51–18:22:28（UTC+8）。

环境：Intel Core i7-14650HX（16物理核/24逻辑CPU）、Ubuntu22.04、GCC11.4的RelWithDebInfo自研程序、Faiss1.12.0 AVX2 wheel与Python3.10.12/NumPy2.2.6。双方单线程固定CPU0（P核），所有阶段快照均为powersave、允许Turbo、接电；实际频率/温度、电源模式及背景负载未完整采样，绑核不等于独占核心。每ef预热1轮、正式3轮、30000个延迟样本，额外10000条查询算严格Recall。

完整SIFT1M的1000000×128 base、全部10000查询、L2、k10、官方groundtruth，M16/Mmax0=32、efConstruction200、seed42，efSearch16/32/64/128/256。

| efSearch | 自研Recall | Faiss Recall | 自研QPS | Faiss QPS | 自研/Faiss |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.802090 | 0.812660 | 14230.4 | 16382.8 | 86.9% |
| 32 | 0.903740 | 0.914430 | 10236.2 | 11201.6 | 91.4% |
| 64 | 0.963770 | 0.970530 | 6661.7 | 6986.6 | 95.3% |
| 128 | 0.989600 | 0.991650 | 3865.6 | 4148.5 | 93.2% |
| 256 | 0.997410 | 0.997990 | 2326.2 | 2205.5 | 105.5% |

默认ef64：自研P99=0.208726ms、Faiss=0.201167ms；构建耗时分别为304.231804110s与351.604918901s。自研严格Recall=0.963770≥0.95，吞吐为本协议Python Faiss的95.3%≥60%；这5个点的同ef吞吐比均超过60%，但自研每点的召回略低，不把比值解释为同准确率下的纯C++性能。没有从Python耗时中估算扣除包装成本；ef256的105.5%也不能证明自研原生内核普遍更快。

自研索引申请容量684722880字节（684.7MB / 653.0MiB）；Faiss有效数组载荷656255596字节（656.3MB / 625.9MiB）。它们口径不同，不据此计算节省比例；本轮没有通过序列化大小冒充内存或进行Faiss剩余容量的完整测量。

仓库保留上述作者测量汇总、参数与环境；CSV、SVG和原始日志只在本地保留。`bench/results/` 与 `data/p2-results/` 都不跟踪，使用者按本文和HNSW评测说明运行，生成自己的结果。取消跟踪不删除本地文件，也不重写既有Git历史。

前文Faiss命令输出 `data/p2-results/faiss.csv`；HNSW评测说明中的扫描命令输出 `data/p2-results/ef-search.csv`。使用相同参数顺序运行后，可生成自己的对照图：

```bash
python3 bench/plot_hnsw_sweep.py \
  data/p2-results/ef-search.csv \
  data/p2-results/faiss-comparison.svg \
  --compare data/p2-results/faiss.csv
```

脚本校验两边工作负载和ef序列，拒绝将多次建图行或Faiss多线程/批量查询结果混入这条对照。Matplotlib只用于可选绘图，Faiss虚拟环境不需要安装它；可用已有matplotlib的系统Python绘图。

本轮收尾集成：rel/san各138项通过，沙盒san临时detect_leaks=0且完整日志无ASan/UBSan告警；用户此前普通终端默认san138项通过记录单独保留。Python8项检查通过。本轮未改C++实现，不要求用户因新增Python工具重复默认san。P2主要单机验收与指定Python基线已达到，取舍和未覆盖项见 [P2验证记录](../docs/validation/p2.md)。
