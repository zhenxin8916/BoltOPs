# Gluon channel GEMM2：短 K WASP persistent

## 当前状态（2026-11-14）

验收口径为 **整个 dispatch、48 CU 平均 MMAC**，包含 HYGPERF 完整 XCD time section 的启动/调度。当前仓库源码直接导入的全量实测为 **2.2 GHz：177.095552 μs / 32.9739%**；**1.75 GHz：179.841024 μs / 40.7487%**。两次均验证 67108864 元素 0 错误。2.2 GHz 相对 4a3677e 的 180.638192 μs 缩短 **1.96%**，仍未在该频率达到 40%；低频成绩单独列出，不能改写成 2.2 GHz 达标。

冻结原型的最快记录仍为 **173.592048 μs / 33.6394%**，候选源码及 patch 已保留。兼容其他短 K / A buffer 路径后的重新编译会改变指令顺序，故此数字不归给当前仓库文件。当前源文件 SHA256 为 `59bcf20688933835bfb58c5b74b22f593eb97b5f9a5af7cb88a19b2fe70bfc16`，本文代码和性能证据一起提交；实际 HSACO/机器指令 SHA256 逐次记录于 `gemm2-20261114/performance-builds.json`。

**统计边界纠正：**此前 177.026395 μs / 32.9868% 来自首 wave 到末 wave 的跨度，漏掉约 3.61 μs 启动时间。旧 `schema_version=1` 的 `time_us` / `whole_gpu_mmac_percent` 以及早期表中“整体”均按该跨度计算；早期 CU0 数字也不能代替 48 CU 平均。`schema_version=2` 改用 perf 完整 dispatch，并将旧值保留为 `wave_span_time_us` / `wave_span_mmac_percent`。下方完整路由主表和 40% 验收表已重算；其余历史前缀/CU0 表保留原值用于比较旧实验，不能用于当前验收。K-body 窗口定义不变，仍只作等待定位。

代码：`boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128_256.py`。探针：`boltops/tools/gluon_moe_gemm2.py`。分支：`dev/gluon-moe-gemm2-20261114`，起点 `24b2089`。初版为 `0b2285e`，各次算法提交的性能分别记录在后文。

真实 GEMM2 为 M4096、E256、topk8、N2048、K256。配置 JSON 文件名的 N256/K2048 是模型中间维度/隐藏维度，不能当作 GEMM2 的乘法维度。BM128 与 GEMM1 一致。CPU 构造输入及 golden，GPU 只运行被测 kernel，结果回传 CPU 检查。

## 算法与展开范围

本节描述 4a3677e 基线；当前 P0 scatter 改动和 barrier 方案见文末。

- P0 搬运 A；P1 依次搬运 Bleft/Bright；C0/C1 分别计算 N 的左右半块。
- K 是 constexpr。生产者、消费者使用 `gl.static_range(K // BK)`：BK128 展开两轮，BK256 展开一轮。persistent 输出遍历保持动态 `range`，barrier epoch 跨输出连续。
- BK128：A/Bleft/Bright 各两槽；BK256：各一槽。BN256 时两者共用 98304 B LDS。A 的 EMPTY 必须等两个消费者都读完才能归还。单物理槽仍交替使用两个逻辑 barrier ID，初始化只注入一个空槽信用。
- scale 与 route weight 在 FP32 累加后相乘，再转换为 BF16，按 route 写到 `[M * topk, N]`；topk reduction 保留在后续算子。

最终 TTGIR 的四个 `scf.for` 都是各 partition 的输出遍历，已没有 K 循环；BK128 的 A 索引 0/128、B 左右半块的 K 索引 0/128 均直接生成。两个消费者各生成 128 条静态 MMAC。汇编 scratch 为 0；BK128/BK256 消费者实际 partition VGPR 分别为 186/222。

## 等工作量诊断结果

PMD sb/gfx946、48 CU、2.2 GHz、HBM 1.8 TB/s。对照为用户指定 gfx938 bottom JSON 的 M4096 配置，实际执行在同一个 sb 模型。所有行处理同一批 12 个排序后 M block、相同 2162688 个有效输出元素，数值错误均为 0。

| 实现 | 算法特点 | BN / BK | 输出 tile 数 | 时间 μs | 整体 MMAC | MMAC 包络利用率 | LDS 冲突 |
|---|---|---|---:|---:|---:|---:|---:|
| 原 Triton JSON | 单分区 persistent、常规加载 | 128 / 128 | 192 | 21.000525 | 8.87% | 12.98% | 1536 |
| Gluon 初版 | 2P2C、A/B MLS、双槽、K 两轮展开 | 256 / 128 | 96 | 13.911170 | 13.40% | 24.59% | 0 |
| BK256 诊断 | 2P2C、A/B MLS、单槽、K 一轮 | 256 / 256 | 96 | 13.741910 | 13.56% | 25.23% | 0 |
| A buffer 诊断 | A buffer-to-LDS、B MLS、双槽 | 256 / 128 | 96 | 14.902615 | 12.51% | 23.63% | 1024 |

BN256/BK128 相对原 Triton 等工作量诊断耗时降低 **33.76%**；包含分块、加载方式、WASP 及展开等全部实现差异，不能归因于单独 unroll。BK256 仅比 BK128 快 1.22%，并增加消费者寄存器需求，暂不据此更改默认 BK128。

A buffer 路径同样数值通过，但比 A MLS 慢 7.13%，且出现 LDS bank conflict；当前默认保留 A MLS。

“整体 MMAC”按四个 SIMD 各自 MMAC 发射的 8-cycle 区间取并集，再除以 dispatch 首 wave 到末 wave 的时间；“包络”分母是各 SIMD 首次至末次 MMAC 的区间，**不是剔除所有等待后的纯 K 热段利用率**。诊断仅每 TG 两个 BN256 输出，启动、scale、scatter 和尾部占比较大，不能将这些百分比当作长 persistent 稳态成绩。

全量真实路由为 48128 个排序后行，即 376 个 BM block：BN256 共 3008 个输出 tile，BN128 共 6016 个。全量验收使用 `--tiles 0`，不能将当前前缀结果外推为完整 M4096 性能。

## Roofline 与后续定位

BM128、K256、BN256 每输出 tile 执行 16777216 FLOP，A/B 共 98304 B，BF16 输出 65536 B。忽略缓存与小型 metadata，计算强度 102.4 FLOP/B，1.8 TB/s 下为 184.32 TFLOPS，约为用户最初提供的 432 TFLOPS 的 42.67%。这不是当前 PMD FP8 指令峰值归一化后的利用率，后文独立校准约为 864.21 TFLOPS。提高整体利用率需要有效的数据复用，只有增加 buffering 不足以保证达到目标。

短 K 展开减少循环维护，但不能消除输出切换及反复加载 A 的成本。下一步需按实际输出轮次拆分等待，检查同一 M block 的多个 N tile 是否能复用 A，同时核对寄存器压力、slot 生命周期及 epilogue，不能用改变统计分母代替性能改善。

## 证据与复现

原始实验根目录：`/home/zhenx/temp/moe-gemm2-20261114/`，分别为：

- `json-baseline-equal-work`：`--mode triton --bn 128 --bk 128 --tiles 192 --grid 48`。
- `initial-bn256-bk128-mlsa`：`--bn 256 --bk 128 --tiles 96 --grid 48`。
- `initial-bn256-bk256-mlsa`：`--bn 256 --bk 256 --tiles 96 --grid 48`。

各目录保留 `launch.py`、`launch.json`（输入 SHA）、`result.json`、`analysis.json`、`gemm/kernel.{ttgir,llir,amdgcn,hsaco}`、原始 IT/stats 及描述性命名 CU0 `.perf`。

运行容器 `zx-perf-node60`；宿主 `/home/zhenx` 对应容器 `/zhenxin`。Python 为 `/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python`。环境设置以各目录 `launch.py` 为准：`setarch x86_64 -R`，CPU 输入固定 seed3101，独立 Triton cache，`GPU_CHIP=sb`、2.2 GHz，StatLog/IT0/TT0/Perf。命令入口 `python -m boltops.tools.gluon_moe_gemm2` 加上述参数。

工具链沿用 GEMM1 的构建，本轮未修改：

- Triton 路径 `/home/zhenx/temp/moe-fullgrid-tune-20261112/mls-bps1/triton`，detached `a3f89b25306c382cb8b01d1bb7d604fffdf63f04`。本地修改 `python/triton/experimental/gluon/language/amd/hcu/__init__.py` 与 `third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp`，包括 MLS BPS1。
- LLVM 分支 `dtk-llvm-6.3.3`，commit `71fabc519b352fb7112d4089abace261e5e177fb`。本地修改 `AMDGPUCodeGenPrepare.cpp`、`SIInsertWaitcnts.cpp` 及其测试，clang wrapper 开启 `-mllvm -hcu-control-only-ebarrier-sync`。未跟踪的 `0001-ds_read-mmerge.patch`、`hcu-matrix-store-lds-waitcnt.mir` 不作为已纳入构建的证明。
- clang18 SHA256：`e3f14dd98d1188838e9f39c825cb546aa20fb25f5b4a79821d743c454cfa2941`。
- PMD `b61a79c5`，`libgem5_opt.so` SHA256：`c45e6168add916ac3c5ae3f4ab06d3d4c245e1970ea91cb09a470b745936040a`。已查阅 PMD 已知问题索引；本次数值通过，不引入模型绕行。

工具链本地 diff 和二进制 SHA 快照另存 `gemm2-20261114/`，与本报告一起版本管理。

## 优化记录：同一行相邻列打包写出

在初版 `0b2285e` 上，仅重排结果寄存器，先排列 N1，再排列 M4/M8；不改变 lane/warp 归属，不改变生产者、消费者、槽位和数值乘法顺序。两个消费者的静态 short store 共 128 条改为 64 条 dword store，exec 分支从 128 降到 64；scratch 仍为 0。消费者实际 VGPR 186 → 187。

同一 96-output 诊断前缀：13.911170 → **12.922910 μs**，耗时降低 **7.10%**；整体 MMAC **14.42%**，包络 **26.43%**，LDS 冲突 0；2162688 个有效输出 0 错误。本节结果属于引入寄存器打包写出优化的提交，记录保存在 `gemm2-20261114/n-contiguous-store/`。

## 优化记录：C 使用 buffer scatter store

在寄存器打包版 `c44953b` 上，仅将 C 的 global store 替换为 scalar C 基址加 uint32 元素偏移的 `buffer_store`。保持 route scatter 语义、有效行 mask、K 展开和流水不变。当前形状输出为 134217728 B，uint32 字节寻址范围足够。汇编生成 64 条静态 `buffer_store_dword`，消费者实际 VGPR 187 → 185，scratch 0。

同一 96-output 诊断前缀：12.922910 → **11.900070 μs**，耗时降低 **7.92%**；整体 MMAC **15.66%**，包络 **29.12%**，LDS 冲突 0；2162688 个有效输出 0 错误。本节结果属于引入 buffer scatter store 的提交，记录保存在 `gemm2-20261114/buffer-scatter-store/`。

Persistent、WASP、WDRA 均启用，2P2C 各 4 waves，requested partition VGPR 为 48/48/208/208。PMD 实际指令 trace 中可见 `s_set_vgpr_size 48` 与 `s_set_vgpr_size 208`，不只依据 metadata 声称启用。非 Gluon JSON 对照生成 8 条静态 `buffer_store_dwordx4`，输出 BlockedLayout 为 sizePerThread=[1,8]、threadsPerWarp=[4,16]、warpsPerCTA=[4,1]；本版只完成 dword store，向量化 x4 仍需测量额外布局转换的成本。

## GEMM1 / GEMM2 的计算量与数据量核对

同一模型 E256、hidden D2048、intermediate F256、M4096、topk8：GEMM1 为 N512/K2048，GEMM2 为 N2048/K256。`prepare_cpu` 的 GEMM1 使用 A[M,D]、B[E,2F,D]；GEMM2 使用 A[M*topk,F]、B[E,D,F]。因此 **GEMM2 的 FLOP 与 B 权重大小都是 GEMM1 的一半，但总 A/B/C 数据量和执行时间不必为一半**。

| 按有效 route 核算 | GEMM1 | GEMM2 |
|---|---:|---:|
| 有效计算量 GFLOP | 68.7195 | 34.3597 |
| 当前 padding 后实际计算量 GFLOP | 100.9317 | 50.4659 |
| 原始 FP8 A footprint MiB | 8 | 8 |
| FP8 B footprint MiB | 256 | 128 |
| 有效 route 的 BF16 C MiB | 32 | 128 |
| 上述 A+B+C footprint MiB | 296 | 264 |
| 432 TFLOPS 下有效计算时间下界 μs | 159.07 | 79.54 |
| 1.8 TB/s 下逐字节一次 HBM 读写的估算时间 μs | 172.43 | 153.79 |

最后一行只是假设全部上述字节恰好各经过一次 HBM 的 cold-data 估算；缓存、padding、重复 tile 访问、读写事务效率都会改变实际流量，不能直接当作 PMD 时间。GEMM2 为 sorted A 分配的当前 padding footprint 是 11.75 MiB；GEMM1 的 sorted C 若保留 padding 是 47 MiB，而不是有效 route 的 32 MiB。topk reduction、量化和通信未计入两列。比较 GEMM1/GEMM2 时间必须使用相同模型、完整路由范围、频率和计时边界，不能拿 48 个诊断输出与全量 GEMM2 混比。

## x4 写出诊断（未采用）

非 Gluon 对照确实使用 `buffer_store_dwordx4`，但 LTS/interleave2 结果每 lane 自然只有相邻两列；得到每 lane 八列需要交换两组 lane/register bit。

| 方案 | 时间 μs | 整体 MMAC | 包络 MMAC | LDS B | 结果 |
|---|---:|---:|---:|---:|---|
| 当前快版 `0ff7c0c`，dword scatter | 11.900070 | 15.66% | 29.12% | 98304 | 已采用 |
| 32-row panel + LDS 转换 + x4 | 14.390740 | 12.95% | 25.61% | 114688 | 未采用 |
| 独立编译器 warp shuffle + x4 | 14.308840 | 13.02% | 23.63% | 98304 | 未采用 |

三行均检查 2162688 个有效输出，0 错误。shuffle 版没有新增 LDS scratch 或 partition barrier，但仍引入两个消费者合计 96 条 `ds_bpermute_b32`、64 条 `v_perm_b32`，抵消了 store 数量从 64 条 dword 减到 16 条 x4 的收益。未把实验编译器改动或两个慢 kernel 纳入正式实现；原始材料在 temp 下 `buffer-scatter-x4-panels`、`buffer-scatter-x4-shuffle`、`lts-store-shuffle`。正式 runtime 仍为本文前述 a3f89b2 + 原有两处本地修改。

## 全量真实路由对照（已完成）

同一 M4096/E256/topk8/N2048/K256、seed3101 路由、48 CU、2.2 GHz。两者完整覆盖同样 376 个排序后 M block、32768 条有效 route，验证 **67108864 个输出元素，均 0 错误**。不是重复同一 tile 的合成 persistent 测试。

| 代码 | 算法特点 | 真实输出 tile | 每 TG 输出数 | 时间 μs | 整体 MMAC | MMAC 包络 | K-body MMAC |
|---|---|---:|---:|---:|---:|---:|---:|
| 原 JSON / 起点24b2089非 Gluon | BN128、BK128、常规加载、单分区 persistent | 6016 | 125/126 | 505.645595 | 11.61% | 11.76% | 未按同一边界提取 |
| 0b2285e 初版 | BN256、BK128、2P2C、A/B MLS、双槽、K 展开、global short scatter | 3008 | 62/63 | 260.294580 | 22.55% | 23.70% | 55.31% |

初版全量耗时降低 **48.52%**。此结果属于 **0b2285e**，不是后续 dword/LDS/x4 优化的全量成绩。完整证据保存在 `gemm2-20261114/full-json-baseline/` 与 `full-bn256-bk128-mlsa/`；perf 和原始 trace 位于同名 temp 目录。

K-body 指各消费者机器指令中“该输出第一个 A READY wait → 最后一条真实 MMAC + 8 cycle”的区间，包含 A/B 等待；对同一 SIMD 的 C0/C1 区间取并集，以真实 MMAC 8-cycle 区间并集为分子。只排除两个消费者同时在 epilogue 等 K-body 外部工作的区间。它与全 dispatch、首末 MMAC 包络不同，三种口径明确分别列出，不用剔除 K 内等待的纯指令密度声称达到目标。

## 优化记录：按 M 持久遍历、A 寄存器复用、LDS x4 写出

本节成绩对应引入本节的实现提交。保持 2P2C、WASP、WDRA，P0 每个 M block 只搬一次完整 K256 的 A；C0/C1 读入寄存器后共同归还 A EMPTY，并保留 A 遍历全部 N tile。P1 按 N 顺序搬 Bleft/Bright。A epoch 按 M 推进，B epoch 按 N 输出推进，互不混用；跨 M 仍通过各自 READY/EMPTY 防止覆盖。K 完全展开，M/N 遍历动态执行。完整 A 预读要求 K 不超过现有 A 槽容量，使用 constexpr 断言防止不支持的长 K 死锁。

输出先在寄存器中配对相邻 N 列并转 BF16，再以两个 64-row panel 经 LDS 转为每 lane 八列，生成 `buffer_store_dwordx4`。只有结果值经过 LDS，route 在输出布局中直接 buffer load；列 mask 由 N%BN 的 constexpr 约束消除，route/expert 有效性仍运行时检查。BN256 输入缓冲 96 KiB，加两个消费者各 16 KiB scratch，合计 **128 KiB**。消费者实际 VGPR 218/217，requested partition 为 16/16/240/240，scratch spill 为 0。LDS bank conflict 尚有 6144，后续仍需优化，不能声称 x4 转换没有代价。

PMD、2.2 GHz、48 CU、同一 seed3101、同一 48 个 M block、384 个 BN256 输出（每 TG 八个），均验证 8974336 个有效元素，0 错误：

| 实现 | 特点 | 时间 μs | 整体 MMAC | MMAC 包络 | K-body MMAC |
|---|---|---:|---:|---:|---:|
| `0ff7c0c` 精确提交源码 | 每 N 重载 A、dword scatter | 35.184695 | 21.19% | 26.61% | 44.45% |
| 后续 mask/ticket 清理诊断 | 每 N 重载 A、dword scatter | 34.183695 | 21.81% | 26.56% | 46.69% |
| LDS x4 + 相邻列打包诊断 | 每 N 重载 A、两 panel | 35.998235 | 20.71% | 25.45% | 见原始分析 |
| 当前 A 复用 + LDS x4 | 每 M 一次 A，八个 N 输出复用 | **29.881670** | **24.95%** | **34.84%** | **57.62%** |

相对精确 `0ff7c0c` 提交耗时降低 **15.07%**；相对清理后的 dword 诊断降低 **12.59%**；相对同样 LDS x4 的非复用诊断降低 **16.99%**。这些是整组实现的收益，不归因于单独 LDS 转换。最新 BN128 的等工作量诊断（768 输出、每 TG 16 个）为 34.280155 μs，K-body 63.20%，但整体较 BN256 慢，未据此切换默认配置。

新算法的 K-body 边界：每个 M 的第一个 N 输出从 A READY wait 开始；后续 N 输出从对应第一个 B READY wait 开始，均截止该输出最后一条 MMAC+8 cycle。保留操作数等待，仅排除 epilogue 独占时段。对应分析脚本随代码保存。

证据：`gemm2-20261114/row-persistent-a-reuse/`、`eight-outputs-0ff7c0c/`、`row-persistent-a-reuse-bn128/`。全量异步运行目录为 `/home/zhenx/temp/moe-gemm2-20261114/row-persistent-a-reuse-full` 与 `full-buffer-scatter-0ff7c0c`，本节不将尚未完成的全量运行记作通过。工具链仍为前述 Triton/LLVM 提交及本地 patch，未使用实验 shuffle 编译器。


## 优化记录：完整 B 读入寄存器后再计算

本节成绩对应引入本节的实现提交，基线为 `da9f08e`。只改变消费者 B 的读入/计算顺序：先等两段 K 的 B READY、读入两个寄存器 tile，`lgkmcnt(0)` 确认 LDS 读取完成后归还两个 EMPTY，再执行原顺序的两段 MMAC。计算开始后不再穿插 B READY 等待，生产者同时可以提前覆盖下一输出的 B 槽。LDS 容量、数值乘法顺序、2P2C 职责及输出散写均保持一致。

同一 2.2 GHz、48 CU、384 输出前缀、8974336 个有效元素（0 错误）：

| 实现 | 时间 μs | 整体 MMAC | 包络 MMAC | K-body MMAC | 实际 C0/C1 VGPR |
|---|---:|---:|---:|---:|---:|
| `da9f08e` A 复用 | 29.881670 | 24.95% | 34.84% | 57.62% | 218 / 217 |
| 完整 B 预读 | **29.082690** | **25.63%** | **36.19%** | **67.47%** | 246 / 245 |

时间降低 **2.67%**，K-body 提高 **9.85 个百分点**。LDS 仍 131072 B、bank conflict 6144，scratch spill 0。WASP requested VGPR 16/16/240/240，实际汇编 WDRA 将两个生产者释放到 8，将消费者增到 248；不能只依据 requested 值判断寄存器容量。K-body 仍包含第一个 B READY 到最后 MMAC 的等待，第一 N 输出还包含 A preload，不改变统计口径。67.47% 是 K-body，不能宣称整体已达到 60%。

证据保存在 `gemm2-20261114/row-a-full-b-preload/`；原始 CU0 perf 为 `/home/zhenx/temp/moe-gemm2-20261114/row-a-full-b-preload/GEMM2_M4096_N2048_K256_BM128_BN256_BK128_GRID48_OUTPUT384_row-a-full-b-preload_CU0.perf`。另在运行 grid4、64 输出的双 M 生命周期检查，仅验证跨 M credit，不作为真实 48 CU 性能。


## 全量验收补充：A 跨 N 复用（da9f08e）

`row-persistent-a-reuse-full` 完整覆盖 3008 输出 tile、376 个 M block、67108864 个有效元素，0 错误；每 CTA 连续处理 7/8 个 M block，验证了跨 M 的 A/B epoch 和槽位生命周期。耗时 **193.274445 μs**、整体 MMAC **30.86%**、K-body **56.10%**。精确 `0ff7c0c` 全量同工作量耗时 **224.062930 μs**、数值同样 0 错误；A 复用版耗时降低 **13.74%**。这是 da9f08e 的完整路由成绩，不能归给后续 24cb205 或 BK256/route 外提版。

另 `row-a-full-b-preload-two-m` 对 24cb205 做 grid4、64 输出（每 CTA 两个 M block）的生命周期检查：1394688 个有效输出 0 错误，READY/EMPTY ID 审计通过；其 45.697470 μs 不属于 48 CU 全量性能。

## 配置优化记录：默认 BK256

本节对应将 kernel 与探针默认 BK 从 128 改为 256 的提交。基线 `24cb205` 已采用完整 B 预读；BK256 把 K256 一次搬入/读入，使用一套物理槽、交替逻辑 ID，减少两段同步和临时寄存器。BM128、BN256、2P2C、A 跨 N 复用和 LDS x4 写出保持一致；BK128 仍可显式选择。

同一 2.2 GHz、48 CU、384 输出、8974336 个有效元素 0 错误：**29.082690 → 28.213640 μs（降低 2.99%）**。整体 MMAC **26.42%**、包络 **35.39%**、K-body **63.42%**；实际消费者 VGPR **222/221**，scratch 0，LDS 131072 B，bank conflict 6144。K-body 比 BK128 的 67.47% 低，但总耗时更短，未只按热点百分比选型。证据在 `gemm2-20261114/row-a-full-b-bk256/`。


## 优化记录：scatter route、mask、行地址在 N 循环外计算

本节对应引入输出 metadata 外提的提交，基线 **903ba04**（BK256）。Sorted route 的读取、CK route 解码、有效行/专家 mask、route*N 行地址只依赖 M；原先每个 N 输出都重复执行，现改为每个 M 计算一次，分别保留两个 panel 的结果。N 循环内只增加列地址、完成原 FP32 乘法顺序、经 LDS 转布局并 buffer_store。没有将运行时 route mask 误当 constexpr。

同一 2.2 GHz、48 CU、384 输出、8974336 元素（0 错误）：**28.213640 → 27.579825 μs，降低 2.25%**；CU0 全 dispatch MMAC **27.03%**、包络 **37.91%**、K-body **63.51%**。消费者实际 VGPR **222/221**，未增加；LDS 131072 B、bank conflict 6144、scratch spill 0。相对 da9f08e 的 29.881670 μs 累计降低 **7.70%**。证据保存在 `gemm2-20261114/row-a-bk256-hoisted-routes/`，新版全量目录 `/home/zhenx/temp/moe-gemm2-20261114/row-a-bk256-hoisted-full` 仍在运行。

独立尝试也说明选型必须同时看时间和热点：

| 诊断方案 | 基线 | 时间 μs | K-body MMAC | 选择 |
|---|---|---:|---:|---|
| BK128、仅 route 外提 | da9f08e | 28.000245 | 51.94% | 总时间更短但 B 等待暴露，不作为默认 |
| BK128、完整 B 预读 + route 外提 | 24cb205 | 28.386540 | 53.84% | 比 BK256 组合慢且热点更低 |
| P0=A+Bleft、P1=Bright | da9f08e | 29.246035 | 63.87% | 比当前默认慢，未混入正式实现 |
| BN128、单个 128-row LDS panel | da9f08e/BN128 | 31.617040 | 49.51% | 比 BN256 慢，未切换默认 |

上述诊断均为同一前缀有效元素数，数值通过。BN128 使用 768 tile 才等于 BN256 的 384 tile 工作量。原始目录分别为 `row-a-hoisted-output-routes`、`row-a-b-preload-hoisted-routes`、`row-a-balanced-producers`、`row-a-bn128-full-panel`。生产者分工组合的进一步测试仍单独保留，不能把未验证组合当作正式成绩。

全量精确基线 `0ff7c0c` 的进一步指标已完成：224.062930 μs，CU0 全 dispatch MMAC 26.20%、包络 27.49%、K-body 42.18%，67108864 元素 0 错误。证据保存于 `gemm2-20261114/full-buffer-scatter-0ff7c0c/`。


## 当前正式版本与生产者分工复核

当前正式代码 **4a3677e**（基线 903ba04）：P0 每 M 仅搬一次 A，P1 搬每 N 的左右 B，C0/C1 留 A 在寄存器、BK256 完整 B 读入后归还槽位，route/valid/row offset 在 N 循环外，输出经 LDS 后 x4 scatter。其 384 输出诊断 **27.579825 μs / CU0 全 dispatch 27.03% / K-body 63.51%**，尚不能替代新版完整路由验收。

在同一前缀上进一步把两个生产者改为 P0=A+Bleft、P1=Bright（同时保留完整 B 预读与 route 外提）：

| 独立组合 | 时间 μs | CU0 全 dispatch | K-body | 数值 |
|---|---:|---:|---:|---|
| BK128、左右 B 分开生产 | 27.659905 | 26.95% | 57.99% | 8974336 元素 0 错误 |
| BK256、左右 B 分开生产 | 28.147665 | 26.48% | 61.27% | 8974336 元素 0 错误 |

均未优于 4a3677e 的时间/热点组合，未替换正式实现。证据保存在同名 `row-a-balanced-hoisted-bk128`、`row-a-balanced-hoisted-bk256` 目录。

当前正式源码另经 `row-a-bk256-hoisted-two-m` 的 grid4、64 输出跨两个 M 生命周期检查：1394688 元素 0 错误，READY/EMPTY 首 ID 非零、无相邻重复 ID 的 trace 审计通过。这是状态复用验证，不作为 48 CU 性能。仍在运行的全量目录：`row-a-full-b-preload-full`（24cb205/BK128）、`row-a-full-b-bk256-full`（903ba04/BK256）、`row-a-bk256-hoisted-full`（4a3677e）。


## 完整路由验收与选型更新

以下均为 **M4096/E256/topk8，实际 N2048/K256，48 CU，2.2 GHz（455 ps/cycle）**；完整覆盖 376 个 M block、32768 个有效 route、67108864 个有效输出元素，数值错误均为 0。旧版按输出 tile 持久遍历，BN256 每 TG 62/63 输出；A 复用版按 M 持久遍历全部 N，每 TG 处理 7/8 个 M，即 56/64 输出。

| commit / 配置 | 算法特点 | 完整 dispatch μs | 48 CU 全 dispatch MMAC | CU0 K-body | 有效吞吐 TFLOPS |
|---|---|---:|---:|---:|---:|
| 起点 JSON / 非 Gluon | BN128/BK128，原始 persistent | 509.257376 | 11.4668% | 未采集 | 67.47 |
| `0ff7c0c` | BN256/BK128，每 N 重载 A，dword scatter | 227.674720 | 25.6486% | 42.18% | 150.92 |
| `da9f08e` | A 跨 N 寄存器复用，LDS x4，BK128 | 196.886240 | 29.6594% | 56.10% | 174.52 |
| `24cb205` | 完整 B 读入寄存器后计算，BK128 | 185.209568 | 31.5293% | 54.03% | 185.52 |
| `903ba04` / BK256 | 完整短 K 搬运，一套物理槽/交替 ID | 185.679584 | 31.4495% | 65.44% | 185.05 |
| **`4a3677e`** | BN256/BK256，route/mask/行地址在 N 外 | 180.638192 | 32.3272% | 56.23% | 190.21 |
| `4a3677e` / BN128/BK128 | 同一源码，较小 N/K tile，全量对照 | 234.851440 | 24.8648% | 未采集 | 146.30 |

有效吞吐只计 32768 条有效 route 的 34.3597 GFLOP；padding 后实际执行 50.4659 GFLOP，不能混算。以上时间/全卡列均按完整 dispatch 重算；`4a3677e` 相对 da9f08e 时间降低 **8.25%**，相对精确 0ff7c0c 降低 **20.66%**，相对原 JSON 降低 **64.53%**。K-body 并非单调上升，前缀成绩不能替代全量选择。903ba04 的全量运行使用 24cb205 源码显式传 BK256，与 903ba04 仅默认参数不同；冻结源码、参数、toolchain 记录均已保存。

数据目录分别为 `row-a-full-b-preload-full`、`row-a-full-b-bk256-full`、`row-a-bk256-hoisted-full`。当前最快全量 perf：`/home/zhenx/temp/moe-gemm2-20261114/row-a-bk256-hoisted-full/GEMM2_M4096_N2048_K256_BM128_BN256_BK256_GRID48_OUTPUT3008_row-a-bk256-hoisted-full_CU0.perf`。

### 48 CU 统计与峰值校准

`model_metrics.py` 验证实际指令族仅为 `v_mmac_f32_16x16x32_fp8_fp8`；所有 CU 的 MMOP 指令合计恰好匹配 padding 工作量。用 `sum(MMOP)*8/(48*4*完整 perf dispatch_cycles)` 计算全卡平均；首末 wave 跨度作为独立历史字段，并对 CU0 核对 `MMOP*8` 等于现有 trace 的 busy 区间并集。完整 BN256 路由为 3080192 条 MMAC；并未用 CU0 处理较多输出的统计代替全卡平均。

独立纯 MMAC 探针（`mmac-throughput-calibration-warm`）：1 TG、4 waves/4 SIMD，动态重复 17 轮，每 wave 每轮 1024 条同族 FP8 MMAC，输出 8192 个 FP32 元素均为 69632，0 错误。冷轮约 45.05%，其余 16 轮每轮 8208 cycle、占用率 **99.805%**；连续发射间隔为 8 cycle。此探针只校准指令吞吐，绝不是实际 GEMM2 的利用率。

按此 PMD 的 48 CU、4 SIMD/CU、每条 16384 FLOP、8-cycle 和 455 ps 时钟线性换算，FP8 理论峰值约 **864.21 TFLOPS**。这与最初用户提供的 432 TFLOPS 数字不是同一个归一化基准；尚未给 432 作硬件规格归因。本文继续保留原 8-cycle 实测定义，不把分母减半制造 60% 达标。按用户的 432/1.8 约为 240 FLOP/B；按当前模型 FP8 指令峰值约为 480 FLOP/B，且 TCC 带宽不能直接当作 HBM 带宽。

### 数据等待归因

`attribute_k_waits.py` 采用与分析器完全相同的逻辑 K-body 窗口，统计 CU0 SIMD0 两个消费者的相邻发射间隔。它是逐 wave 的时间归因，指令有重叠，**不是额外的硬件 busy 计数，不能跨 wave 直接相加**。

| 全量版本 | C0 Bleft READY 间隔 cycle | C1 Bright READY 间隔 cycle | C0 A READY 间隔 cycle |
|---|---:|---:|---:|
| BK256、未外提 route（903ba04 配置） | 16608 | 496 | 32 |
| 4a3677e、外提 route | **30256** | 384 | 32 |

C0 的 Bleft 等待增加明显，而 C1 的 Bright 几乎始终 ready；A 不是这组 full-run 的主要 READY 等待。由此把下一步集中于左右 B 共同生产者的队首阻塞，以及 LDS/写出反压。正在独立做 P0=A+Bleft、P1=Bright 的 BK128/BK256 全量比较，不依据前缀表现直接采用。

补充 model 计数：0ff7c0c → da9f08e 全量时，平均每 CU 的 GlobalMemPipeline.cmdCreditStall 从 29826.8 降到 10640.8，addrCreditStall 从 12400.3 降到 1803.6；但 LocalMemPipeline.creditStall 从 981.1 增到 19778.1。说明压力从全局命令/地址部分转向 LDS 转换，不能只看一种 credit。da9f08e 的 TCC send response 带宽 1.787 TB/s，SQCI hit rate 99.9937%；这些分别是 TCC 指标与缓存命中率，不能据此直接宣称 HBM 已满。

### 输出转换诊断（尚未替换正式实现）

同一 384 输出前缀，均验证 8974336 个有效元素 0 错误。当前 LDS 容量仍为输入 96 KiB + 两消费者 scratch 各 16 KiB，完全占用 128 KiB。

| 独立尝试 | 时间 μs | CU0 K-body | LDS conflict | 结果 |
|---|---:|---:|---:|---|
| 4a3677e 基线 | 27.579825 | 63.51% | 6144 | 正式版本 |
| 显式 LDS swizzle (8,1,8) | 27.520220 | 60.11% | 4096 | 小收益，未替换 |
| 显式 LDS swizzle (8,4,4) | 27.340040 | 64.61% | 4096 | 小收益，继续 full-run |
| 显式 LDS swizzle (1,1,8) | 33.365150 | 57.07% | 16384 | 排除；静态单指令 conflict 小不等于整核快 |
| 消费者 warp 分布 [4,1] | 27.739530 | 见 temp 分析 | 8192 | 未优于默认，不采用 |
| 输出 warp 分布匹配 MMA（N32/M16） | 27.079325 | 63.05% | 4096 | scoped barrier 30→26，继续 full-run |
| N 列偏移移入 scalar C descriptor | 27.599845 | 63.47% | 6144 | 无收益，不采用 |
| 上项 + 输出 warp 分布匹配 | 27.462435 | 62.82% | 4096 | 比仅改 warp 布局慢，不采用 |
| dword scatter 对照、保留 A 复用/BK256 | 27.558895 | 64.15% | 0 | 与基线基本相同；保留用户要求的 LDS x4 路径 |

这些实验没有修改 LLVM/Triton 工具链，也没有新模型绕行。全部临时代码和原始 trace 位于对应 temp 目录，不将无收益代码混入正式 kernel。纯 MMAC 冷/热测试只说明冷启动指令供应会影响首轮；真实 4a3677e 前缀中，C0 第一个 N 的 128 条 MMAC 跨度为 3860 cycle，后续七个 N 均为 1024 cycle、没有内部发射空档，继续缩短 MMAC 序列自身不是当前主要方向。

## 40% 全 dispatch 验收：生产者分工与输出布局的全量复测

用户明确选择整个 dispatch 的 48 CU 平均，并将目标更新为 40%。保持相同路由、padding 与 MMAC 总数时，需从 180.638192 μs 缩短到 **145.988267 μs**，即至少降低 **19.18%**；不能用 K-body 指标替代。

以下全部为 2.2 GHz、完整 3008 输出、67108864 元素，数值错误均为 0。实验以冻结的 4a3677e 源码为起点，分别只改生产者分工或输出转换，不合并变量。

| 冻结实验目录 | 算法特点 | 完整 dispatch μs | 48 CU 全 dispatch MMAC | CU0 K-body |
|---|---|---:|---:|---:|
| `row-a-bk256-hoisted-full` | 正式版；P0=A，P1=Bleft/Bright | 180.638192 | 32.3272% | 56.23% |
| `row-a-balanced-hoisted-bk128-full` | P0=A+Bleft，P1=Bright；BK128 | 182.673408 | 31.9671% | 65.70% |
| `row-a-balanced-hoisted-bk256-full` | P0=A+Bleft，P1=Bright；BK256 | 186.107744 | 31.3772% | 65.73% |
| `explicit-lds-swizzle-8-4-4-48cu-full` | 显式 LDS swizzle (8,4,4) | 179.835568 | 32.4715% | 59.22% |
| `output-layout-warp-local-full` | 输出 warp 布局匹配 MMA，scoped barrier 30→26 | 180.869328 | 32.2859% | 63.83% |

显式 swizzle 只缩短 0.44%，完整 dispatch 提高 0.1443 个百分点；前缀中输出 warp 布局的收益没有在全量复现。均未达到本轮提交优化的收益门槛，不替换正式 kernel。

分开生产左右 B 确实解决了所怀疑的队首阻塞：CU0 SIMD0 的 C0 Bleft READY 发射间隔，从正式版 30256 cycle 降到 BK128 的 528、BK256 的 432 cycle；C1 的 Bright READY 分别为 640、592 cycle。K-body 因此升到约 65.7%，但整体反而更慢。这组对照证明：**局部 B READY 等待不是整体利用率的唯一决定因素**，不能据此直接采用生产者分工改动。归因仍是逐 wave 的相邻发射间隔，不能跨 wave 相加当作硬件 busy。

### 外存流量与可行性核对

本次进一步核对实际 `config.ini` 与 PMD 配置代码：`system.umcs` 为 `UmcSimulate`，`rw_credit_share=true`，`umc_credit=88`，时钟周期 2276 ps；运行参数明确为 `--umc-bw=1.8TB/s`、`--TCCBlockSize=256`、`--EABusSize=256`。PMD `configs/gpu.py::updateUmcConfig` 明确以 “read/write use the same block” 计算 UMC 时钟；因此不能假定读写各自有完整 1.8 TB/s。

4a3677e 的 UMC 原始计数为 614930 次读、515390 次写。若按每次占用一个 256 B 总线事务估算，共 289361920 B，对应完整 dispatch 内约 **1.6019 TB/s**，已接近设定带宽；同样流量的理想时间约 160.76 μs，仍高于 40% 对应的 145.99 μs。此处是明确带事务粒度假设的流量估算，尚未独立验证每类请求有效载荷，也没有把 TCC response 带宽冒充 HBM 字节计数。

当前 sorted A footprint 为 11.75 MiB，完整 FP8 B 为 128 MiB，有效 BF16 C 为 128 MiB。若这些字节各经过 HBM 一次，约 155.98 μs；缓存初始状态、驻留与写回时机可能改变 dispatch 内流量，所以这只是 cold-data 估算，**尚不据此宣称 40% 不可能**。接下来需优先核对外存事务/缓存复用和 epilogue，而不是只增加 buffering 或再提高局部 K-body。

### 公共语义回归

冻结正式源码做 grid4、每 CTA 两个 M block 的生命周期验证；这些是语义检查，不作为 48 CU 性能数据：

- `regression-a-buffer-bk256`：A 用 buffer-LDS，1394688 元素 0 错误。
- `regression-bn128-bk128`：BN128/BK128，1394688 元素 0 错误，LDS 81920 B。
- `regression-flat-route`：普通 route 编码（CK_SORTING=False），1394688 元素 0 错误。
- `regression-nonlocal-experts`：包含 expert=-1，预置零输出保持不变，1394688 元素 0 错误。
- `regression-bn128-bk256`：默认调度下编译器报告 SGPR spill，被 WDRA 检查拒绝，未进入数值测试。随后 `regression-bn128-bk256-ilp` 仅设置 `amdgpu_enable_max_ilp_scheduling_strategy=True`，通过正常 WDRA 检查、1394688 元素数值验证，以及 abarrier 首 ID 非零/无相邻重复审计。此配置需要调用者显式使用该调度选项；没有全局更改默认选项，也未绕过 spill 检查。其 grid4 时间不用于正式性能对比。

输出 streaming hint 的独立诊断 `output-streaming-store-cs`：只给 `buffer_store` 增加 `cache=".cs"`。当前工具链仍生成 aux=0 的 `llvm.amdgcn.raw.ptr.buffer.store` 和无额外 cache flag 的 `buffer_store_dwordx4`；384 输出前缀为 27.579825 μs，与基线一致，8974336 元素 0 错误。该 hint 在当前生成代码中没有生效，不采用，也不把它当作已测试硬件 streaming store 的结果。

## B 预读与 UMC credit 的进一步隔离

### 下一输出 B 先于当前 scatter 读取：排除

`b-next-before-scatter-two-m` / `b-next-before-scatter-48cu`：当前 MMAC 的结果先完成原顺序 FP32 缩放并转为 BF16，再将下一 N 输出的完整 B 读入寄存器、等待 LDS 完成并归还其 EMPTY，最后进行当前输出的 LDS 转换与 buffer scatter。A 仍跨 N 保存在寄存器；所有 B slot、phase、READY/EMPTY 次序保持原算法，不改生产者。

| 同范围对照（本历史表为 wave 跨度） | 原版 μs | B 预读 μs | 数值 |
|---|---:|---:|---|
| grid4、每 CTA 两个 M | 41.046005 | 45.471790 | 1394688 元素 0 错误 |
| 48 CU、384 输出 | 27.579825 | 28.423850 | 8974336 元素 0 错误 |

上述历史前缀的 wave 跨度 MMAC 从 27.0296% 降到 26.2270%；不作为完整 dispatch 验收值。消费者实际 VGPR 222/221→250/249，编译器将 WDRA 配额由 16/16/240/240 自动调整为 4/4/252/252，无 spill，LDS 仍为 131072 B。READY 前移也让 K-body 窗口跨过上一个输出的 scatter，不能再将该窗口百分比直接与原版比较。寄存器压力和耗时均增加，不进行无依据的全量重跑，不替换正式实现。

### TCC 实际容量：6 MiB

`tcc-constructor-audit` 使用诊断用 ELF audit 代理，原样转发 TCC/TagStore 构造函数参数与返回，不修改模型配置、指令或同步。固定模型库 SHA256 为 `c45e6168add916ac3c5ae3f4ab06d3d4c245e1970ea91cb09a470b745936040a`。日志确实捕获 **32 次 TagStore 构造调用，每次参数 (48,16,0,8)**，与 config 的 32 个 TCC、48 sets、16 ways、256 B/cache line 一致。

同一二进制中 `TagStore::TagStore` 按第一个参数创建 set，向 `BaseSet` 传第二个参数作为 ways；不是再乘一次 32 个物理 bank。因此总容量为 **32×48×16×256 = 6291456 B，即 6 MiB**。代理运行的 262144 元素验证为 0 错误，仅证明观测路径正常；不把带代理的运行当作正式性能比较。源码、容量 JSON 和构造日志摘要一并保存。

### 原生 UMC 日志：credit 共享与事件语义

`umc-native-trace-two-m` 只增加 PMD 原生 UMC/MemSim 日志，kernel 未改。grid4、两个 M 的首末 wave 跨度仍为 **41.046005 μs**，完整 dispatch 为 **44.657796 μs**，1394688 元素 0 错误。日志共 16961 个唯一 packet ID，恰好等于 UMC 的 12028 次读 + 4933 次写。

必须区分日志字面与模型执行位置：字符串 `send response od pkt` 的 DPRINTF 位于 `UmcSimulate::sendResp` 将请求放入延迟队列的位置；之后另一段循环才检查成熟时间并调用 `RespPort::sendResp`。所以脚本统计的是 **UMC 入队/服务开始事件，不是完成、credit 归还，更不是消费者 B READY**。即使所有此类日志都早于最后一个 wave，也不能据此宣称所有 HBM 请求已经返回。

模型每组有两个读/写端口，共 32 个端口；`rw_credit_share=true` 是共享 credit，并不意味着每组每周期只能入队一个请求。小范围原生日志的单周期峰值为 17，确实可能超过 16。真正的共享限制是 **16 组×88 credit = 1408 个在途额度，最短驻留时间 87 个 UMC cycle（2276 ps/cycle）**；应检查整个驻留窗口，不能用单周期合计大于 16 判断模型违规。

`umc-request-kind-audit` 另以只读字段代理关联请求种类与原生日志 ID，16884 个请求 ID 与 16884 个原生日志 ID 完全一致；命令码 1/2 合计 12003，恰好匹配读计数，72 为 4881，恰好匹配写计数。每周期读峰值 16、写峰值 14。该代理只读当前版本反汇编确认的 packet 字段；不使用不安全的跨库 `Packet::print` 探针，后者诊断失败不能归因于 PMD 数值/流水缺陷。带代理的数据不作为正式速度对比。

新增 `umc_response_window.py` 同时检查 packet 唯一性、R+W 总数、UMC 时钟相位、端口上限、87-cycle 窗口的 1408-credit 上限，并从所有 48 CU 的 MMOP 与真实 GC 时钟计算全 dispatch 利用率。对于未输出 IT0 的运行，仍核对最终 FP8 指令族和 padded 工作总量，不伪造 IT trace 或 K-body 结果。

全量 native UMC 对照和 BN128/BK128 对照已完成，结果见下一节。

## 完整 dispatch 边界复核与全量 UMC 结果

`perf_dispatch_span.py` 直接解析 HYGPERF 的 XCD time section（type15、72 B；布局与当前模型的 `PerfWriter::OutputXCDTimeSection` 一致）。正式版 perf 给出 begin=0、end=180638192 tick，与 `totalSimTicks=180638185` 只差 7 ps 的浮点输出舍入。首 wave=3611790 tick，因此旧 177026395-tick 跨度漏掉约 3.61 μs。每次测量还检查只有一个 kernel、48 CU 的 MMOP 总数恰好匹配同一 50.4659 GFLOP，以及 FP8 指令族为 16×16×32。峰值/8-cycle 校准和数学工作量没有改变。

两个完整测试均验证 67108864 元素 0 错误；输入全部 SHA256 完全一致，sorted rows 同为 48128。原生日志没有采集 IT/TT 文本，故这两次仅提供完整 dispatch/事务结果，不新增 K-body 或 wavefront timeline。

| 配置，同一 4a3677e 源码 | 完整 dispatch μs | 48 CU 全 dispatch MMAC | UMC 读 | UMC 写 | 87-cycle 窗口峰值/1408 | 全 dispatch credit 驻留占比 |
|---|---:|---:|---:|---:|---:|---:|
| BN256/BK256，native UMC 复测 | 180.638192 | 32.3272% | 614930 | 515390 | 1407/1408 | 88.00% |
| BN128/BK128，全量对照 | 234.851440 | 24.8648% | 760979 | 721748 | 1408/1408 | 88.79% |

较小配置的入队事务合计增加约 31.18%，完整耗时增加约 30.01%；两者 credit 驻留占比接近，峰值均到容量附近。这支持把主要精力放在外存事务效率。BN128 时每个消费者只写 64 个 BF16 列，即 128 B，为 256 B cache line 的半行；两消费者错开发送可能损害合并效率。但该对照同时改变 BN/BK，不能把全部差异直接归因于 N 或 C 半行写回；B 的 K128 分段搬运也可能影响合并。相同 BK128 下的 BN256 控制与左右 C 相邻写出控制现已完成，见下表。

TCC 容量计算还核对了当前二进制的 `TCCBank::extractSetTag`/`getBlkAddr`：tag 以 `addr >> block_offset_bits` 标识，反向地址是 `tag << block_offset_bits`，没有把一个 tag 扩为 32 条 line。结合 32 次 (48 sets,16 ways) 的 TagStore 构造记录和 256 B line，确认 6 MiB；不是将物理 bank 数再乘一次。

## P0 接管 scatter 的 2P2C 优化（完整候选验证通过，合入版复测中）

原算法中 P0 搬完 A 后，大部分 N 循环时间空闲。原型让消费者保持原 FP32 缩放顺序，将 BF16 结果按两个 64-row panel 写入每 C 一个 LDS mailbox；P0 读出 C0/C1 后归还 mailbox，再相邻发两部分 buffer store。消费者可以在 P0 写出时继续 MMAC。P0 同时在当前 M 开始后预取下一 M 的 A，沿用“两个消费者读完 A 后才归还输入 EMPTY”的条件。BN256 时每 C mailbox 为 16 KiB，输入 96 KiB + mailbox32 KiB = 128 KiB；BN128 时输入 64 KiB + mailbox16 KiB = 80 KiB。仍是 2P2C、WASP+WDRA。

第一版共同 READY 等待两个消费者，第二版拆开输出交接，避免用写出同步抹掉 C0/C1 的自然错位。第二版限定 K=BK256：每半 B 只有一物理槽，消费者的输出 mailbox 等待已经分隔了连续的 B 等待，所以 B 的每半 READY/EMPTY 各可只用一个 ID；P1 则交替等待左右 B EMPTY。由此腾出 ID，在 16 个硬件 ID 内独立交接 C0/C1。

| 第二版对象 | READY | EMPTY | READY/EMPTY 到达 wave 数 |
|---|---|---|---|
| A | 1/2 | 3/4 | 4 / 8 |
| Bleft | 5 | 6 | 4 / 4 |
| Bright | 7 | 8 | 4 / 4 |
| C0 mailbox | 9/10 | 11/12 | 4 / 4 |
| C1 mailbox | 13/14 | 15/0 | 4 / 4 |

ID0 只用于 C1 后续的第二 panel EMPTY；所有 wave 的第一个 try_wait 仍是非零 A/B ID。每个 mailbox 只有一块物理存储、两套交替逻辑 ID/phase；消费者写 LDS 并等待 LGKM 完成后才发 READY，P0 读完 LDS 并等待 LGKM 后才发 EMPTY。P0 将数据留在私有 VGPR，再发 global store，因此归还 mailbox 后覆盖 LDS 不会破坏待写出的值。跨两个 M 的数值与“首等待非零、相邻等待 ID 不重复”审计均通过，没有扩大 ID 范围或复用未退役的 lifetime。

以下是匹配范围的局部对照，时间均按完整 perf dispatch 重算，**不把 grid4 或 384 输出当成完整路由验收**。

| 算法 / 范围 | 完整 dispatch μs | 数值 |
|---|---:|---|
| 4a3677e，BN256/BK256，48 CU/384 输出 | 31.191616 | 8974336 元素 0 错误 |
| P0 scatter，共同输出 READY，BN256/BK256，48 CU/384 输出 | 31.917340 | 8974336 元素 0 错误 |
| P0 scatter，独立交接，BN256/BK256，48 CU/384 输出 | 30.159220 | 8974336 元素 0 错误 |
| P0 scatter，共同输出 READY，BN128/BK128，48 CU/768 输出 | 31.479630 | 8974336 元素 0 错误 |
| P0 scatter，BN128/BK128，store布局适配 CN64，48 CU/768 输出 | 29.832984 | 8974336 元素 0 错误 |

独立交接版相对同范围基线缩短约 3.31%。BN128 的另一个单独变化仅将 P0 store 的 threadsPerWarp 从 [4,16] 改为 [8,8]，sizePerThread 仍为 [1,8]；这与 CN64 相匹配，避免 128-column 分布在 64-column panel 上的冗余 lane，mailbox 与输入协议保持不变。其全量验证为 175.210496 μs / 33.3287%，慢于 BN256 独立交接版，因此正式默认路径不选它。

完整路由全量结果已完成，均为 48 CU / 2.2 GHz，67108864 元素 0 错误。下表全部采用完整 XCD dispatch 时间，不能与旧首末 wave 跨度混用。

| 算法 | BN/BK | 完整 dispatch μs | 48 CU 全 dispatch MMAC | 相对 4a3677e 时间降低 |
|---|---|---:|---:|---:|
| 正式基线 4a3677e | 256/256 | 180.638192 | 32.3272% | — |
| 同 BK128 的 BN256 控制 | 256/128 | 179.396032 | 32.5511% | 0.69% |
| P0 scatter，共同 READY | 128/128 | 184.267264 | 31.6905% | -2.01% |
| P0 scatter，共同 READY，store64 布局 | 128/128 | 175.210496 | 33.3287% | 3.00% |
| **P0 scatter，C0/C1 独立交接** | **256/256** | **173.592048** | **33.6394%** | **3.90%** |

BN256/BK128 的控制说明 BN128/BK128 的原始退化不能归因于 BK128 本身。P0 相邻写出也把 BN128 的原始 234.851440 μs 降到 184.267264 μs，store 布局再降到 175.210496 μs；但最佳仍是 BN256 独立 mailbox 交接。它让 P0 兼任 A producer 与 scatter writer、P1 继续搬 B，消费者保留 A 并进行计算/缩放与 LDS 输出，仍为 2P2C。

正式文件现已加入 K=BK=BN256 且 MLS A 的独立交接路径，其余短 K / A buffer 配置保留原 consumer scatter。寄存器默认值按 constexpr 分支选择，显式 PARTITION_REGS 仍由调用方指定。合入后的两个 M 测试和 BK128 fallback 都验证 1394688 元素 0 错误；独立路径 barrier 审计通过。因为合入时源码位置和 constexpr 分支影响了编译后的寄存器分配，**173.592048 μs 只归属于冻结候选源码**，不能冒充合入版已经复测过的性能。合入初版 `p0-scatter-integrated-full` 已完整数值通过，176.145968 μs / 33.1517%。其后补测发现输出交接路径配 A buffer 时超出 VGPR 配额，故最终源码只在 MLS A 启用该路径；A buffer 回到原分支后，两 M 的 1394688 元素验证通过。这个最终分支条件也改变了机器码。冻结目录的 final2 全量为 177.793072 μs / 32.8445%，随后增加普通仓库导入的完整控制，成绩以本文开头及下表为准。

候选完整 perf：[独立交接版 173.592048 μs](/home/zhenx/temp/moe-gemm2-20261114/p0-a-and-output-scatter-independent-full/GEMM2_M4096_N2048_K256_BM128_BN256_BK256_GRID48_OUTPUT3008_p0-a-and-output-scatter-independent-full_CU0.perf)。统计、冻结源码相对基线的 patch、编译元数据、数值结果与 perf SHA256 已归档到本报告同目录。


A padding 另做了两个分开的控制：A buffer-LDS 的 48 CU 前缀为 31.852730 μs；再加 P0 行 mask 为 32.185336 μs，均验证 8974336 元素 0 错误，慢于 MLSA 的 31.191616 μs。mask 仅跳过无效 sorted A 行，route decode 在 K 搬运前完成；寄存器需求增加，未采用。LLVM/Triton/PMD 构建保持此前记录，新增的是候选源码和诊断/统计脚本。

## 本轮发射压力与寄存器控制

| 完整路由 | 原 4a3677e | 独立输出交接候选 |
|---|---:|---:|
| UMC 读请求 | 614930 | 606358 |
| UMC 写请求 | 515390 | 513102 |
| UMC credit 驻留占比 | 88.00% | 90.69% |
| GlobalMemPipeline.cmdCreditStall，48 CU 均值 | 19359.81 | 31491.40 |
| LocalMemPipeline.creditStall，48 CU 均值 | 27326.13 | 19367.94 |

P0 写出确实提高了内外流水的重叠，但全局命令 credit 等待增加。计数器是流水阻塞周期，不能相加当成 dispatch 空档时间，也不能仅凭命令 credit stall 判定 L1 miss。后续需同时看外存事务、credit 压力与发射/消费次序。

把下一 N 的 B 读取插到 C 输出两 panel 之间，BN256 编译反馈超出 VGPR 总配额，尚无可比较的性能；BN128 可通过两个 M 的数值和等待 ID 审计，但它不是同 BN 的单变量速度结论。另将独立交接的 BN128 P0 store 改为 CN64 对应布局时，编译器报告 SGPR spill，WDRA 不允许，未运行数值或记为性能结果。没有修改 LLVM/模型或绕过硬件约束来掩盖这些失败。优化主路径因此限定已验证的 BN256，BN128 继续使用原短 K 分支。

合入初版的分支条件只为 BN256 启用独立输出交接。该阶段完整测试 HSACO 与当时源码重新编译的 HSACO，ELF `.text` SHA256 都为 `7e7fef66606d67be0f23b12283a3e194a5a0d50b08adf4c6f3aed9a937c1b1c4`，逐字节一致；证据为 `gemm2-20261114/integrated-final-source-equivalence.json`。该记录仅对应合入初版；之后增加 MLS A 条件在 compile-only case 得到 `2dde9c9853d91238ff32a5c5b4ba2f8a55958cee4dbd63cbdf5f4ab3f73392bd`；冻结 full case 和普通仓库导入的机器码又不同，均记录在 performance-builds.json，不能把 compile-only hash 冒充每次实际执行的 hash。

## 合入初版全量结果与后续单变量控制

合入初版完整覆盖 67108864 元素，0 错误，时间 **176.145968 μs**、48 CU 全 dispatch MMAC **33.1517%**，比原 4a3677e 缩短 **2.49%**。原始 perf 与完整计数已归档于 `gemm2-20261114/p0-scatter-integrated-full/`。它慢于冻结候选 173.592048 μs：UMC 读从 606358 增到 623440（+2.82%），写从 513102 增到 513737；credit 驻留占比仍几乎相同，90.69% / 90.79%。不能仅凭同一算法断言性能相同。

LLVM IR 对比还确认初始 A 的两组 16 KiB MLS 发射次序发生重排，地址 hoist 与寄存器分配也变化。这是编译生成时序差异的证据，尚未以单变量控制证明全部退化由这一个重排导致；增加的 UMC 请求与耗时变化相符。

| 48 CU / 相同 384 输出前缀 | 完整 dispatch μs | 48 CU 全 dispatch MMAC | 结论 |
|---|---:|---:|---|
| 合入初版同源码控制 | 29.993144 | 24.8547% | 控制组 |
| 只改 mailbox swizzle (8,4,4) → (8,1,8) | 30.019080 | 24.8333% | 无收益，不采用 |
| 只设 P0 priority1、结束恢复0 | 30.741620 | 24.2496% | 变慢，不采用 |
| B 预读，剥离最后 N 输出解除 VGPR 超限 | 32.818696 | 22.7149% | 慢于冻结候选 30.159220 μs，不采用 |

所有前缀均检查 8974336 元素，0 错误。前缀、grid4 和全量数字分开使用。B 预读剥尾版最终配额为 68/12/216/216，实际需求 55/2/212/211，未绕过 512 VGPR 总预算；两个 M 的等待 ID 审计通过，但资源合法不代表性能受益。

## UMC 按 A/B/C 归类：只读诊断

`umc-address-kind-vmap-two-m` 在相同 kernel 外增加只读 ELF audit：原样调用 `getReqPkt`，读取当前模型 Packet 的 cmd(0xc)、id(0x10)、addr(0x50)、size(0x5c)。地址和 size 偏移由同一模型库 `Packet::getAddrRange` / `matchAddr` 反汇编确认；没有跨库调用 C++ 字符串/打印函数。GPU tensor virtual range 来自探针 manifest，物理 range 由原生 `UTCL2_PAGETABLE` 的 2 MiB PDE0 映射关联，不假设低位地址截断。

两个 M、1394688 元素 0 错误；16575 个 packet ID 与原生 UMC admission ID 完全一致。A 为 1024 个不同 256 B line，B 为 10266 请求 / 10240 不同 line，C 为 4627 个不同 line 的 dirty writeback；scale/route/权重、code 与 page-table 请求分别保留，未把未知地址硬归到 B。小范围 C 仍有部分驻留缓存，不能据此外推完整路由的写出字节。

A/B/metadata 的 read packet header size 为 64 B，code 部分为 32 B，而 C dirty writeback 为 256 B。数据 read 的物理地址按 256 B line 分布；**header size 既不能直接当作实际总线 payload，也不能默认每一种请求都搬 256 B**。此前乘 256 的流量只保留为有明确假设的估算。完整 `umc-address-kind-vmap-full` 已启动，以区分 B 重复读取与输出写回，不使用带 audit 的运行时间作为正式性能。

额外构造转发诊断观察到 32 次 TCC TagStore(48,16,0,8)、48 次 TCP TagStore(4,128,0,0)，262144 元素 0 错误；原始参数与 build 命令已保存。TCP tuple 与 config.num_ways 不同，尚未把该 tuple 直接换算成有效 byte 容量，更未据此判断模型缺陷。LLVM/Triton/PMD 构建没有修改。

同一源码分别重新编译的低频测量也已完成，见下表。源码相同不保证 HSACO 字节相同；因此额外启动固定同一 HSACO 与相同 GPU 地址的频率控制，避免把代码生成差异全部归因于频率。

## 本文提交的仓库文件：全量复核

下面两次使用 `python -m boltops.tools.gluon_moe_gemm2 --bn 256 --bk 256 --tiles 0` 直接导入仓库，未覆盖 Python module。探针现在记录 kernel source SHA256 和各 GPU buffer 的地址/字节数；这些是 CPU 侧溯源记录，不增加被测 GPU 工作。两个测试都是同一完整路由/48 CU，67108864 元素 0 错误。

| 源码与运行方式 | 时钟 GHz | 完整 dispatch μs | 48 CU 平均 MMAC | 有效 TFLOPS |
|---|---:|---:|---:|---:|
| 本提交，普通仓库导入 | 2.2 | **177.095552** | **32.9739%** | 194.02 |
| 本提交，普通仓库导入 | 1.75 | **179.841024** | **40.7487%** | 191.06 |
| 同源码，冻结目录 final2 | 2.2 | 177.793072 | 32.8445% | 193.26 |
| 同源码，冻结目录 final2 | 1.75 | 181.854928 | 40.2975% | 188.94 |
| 较早合入候选，冻结目录 | 2.2 | 176.145968 | 33.1517% | 195.06 |
| 较早合入候选，冻结目录 | 1.8 | 177.712064 | 40.1536% | 193.34 |

各次独立编译的 HSACO 不同，完整频率因果比较必须固定二进制。低频利用率的分母采用该次实际 GC period，不是继续用 2.2 GHz 周期计算。1.75 GHz / 40.75% 是真实低频点，不是 2.2 GHz / 40%。用户关于是否要求最高频达到 40% 的口径问题仍待明确，继续保留最高频优化任务。

普通仓库导入的 2.2 GHz HSACO SHA256 为 `a19432c132bc6c28e7a5c5dc2e9b41ea92d8261c9f00139cedcba01ec028175f`，`.text` 为 `5122a7f78577961bbb556c774dc984f9deaa924f09384ac631c6e8e34771811a`。已用该 cache、`TRITON_ALWAYS_COMPILE=0` 追加 1.8 / 1.75 GHz；启动后检查两者整个 HSACO SHA256、所有输入 SHA256、所有 GPU buffer 地址和字节数均与 2.2 GHz 控制完全相同，完整数值与计数已完成，见下方固定二进制对照。

上述普通仓库和 final2 的 perf 仅采 StatLog/Perf/UMC，提供完整 dispatch 和全卡计数，没有新增 SQTT wavefront trace。先前合入初版的完整 IT/TT 可作该版本的指令诊断，但不能冒充当前二进制的 wavefront。

## 全量 UMC 地址归类与下一个瓶颈

只读 audit 全量运行检查 67108864 元素，0 错误；1134722 个 request ID 与原生 UMC admission ID 精确一致。物理地址来自原生 PDE0 翻译，独立于 tensor 虚拟地址布局。结果是：

| 类别 | 请求数 | 不同 256 B line | 重复 line 请求 |
|---|---:|---:|---:|
| A read | 48128 | 48128 | 0 |
| B read | 556895 | 524288 | 32607 |
| C dirty writeback | 514812 | 514812 | 0 |
| route weight read | 4637 | 512 | 4125 |
| SB read | 8653 | 8192 | 461 |

该运行没有观测到 C read，不支持把主要额外读流量解释成 C 的 read-for-ownership。A/B/C 占据绝大部分请求，B 存在约 3.26 万个重复 line 请求；不能把 pipeline stall 数全部当作 B 数据没回来。Audit 改变分配和时序，不用其耗时作为正式成绩，也不把 header size 与实际总线 payload 混为一谈。

## 两 TG 驻留设计与 buffer offset 的独立定位

两 wave/partition 的设想被当前 WDRA 检查拒绝；没有绕过硬件限制。保留每 partition 四 waves、改为 consumer warp grid [4,1] 后，64 KiB 原型的数值通过，但编译器将请求的 256 总配额扩成 512（实际需求 38/2/121/121），所以尚不能声称双驻留。Coalesced SB 转布局反而增加寄存器并额外申请 512 B LDS，未采用。

另一个 A 留 LDS、B 单槽原型出现 expert0 前七个 N 块一半列为零；单 M、单 TG 对照仍复现，已排除下一 M A 预取覆盖的猜测。IT 直接显示 Bright 的 SB `buffer_load_dwordx2` 使用 `VOFFSET=0xfffff200`、imm3840，加载返回零。LLVM LSR dump 先把 IV rebasing 到 -7680，再形成 i32 add7936；后续 buffer offset split 把它变成 VGPR -3584 与 imm3840。原 i32 参数应为 256，而拆开后的硬件范围检查不是同一个 modulo-32 表达式。

独立 probe 只读取同一普通数组，比较正常 buffer_load、显式拆分和先做完整 u32 加法再 load 三列。正常 load 与完整 u32 加法均为 16/16 正确；拆分列恰好在 n0..6 返回零，n7..15 正确，完全匹配 kernel 的失败范围。LLVM `splitBufferOffsets` 自身也注明不能把负 VGPR offset 留给立即数补正。故下一步核对并修复编译器 offset 拆分的 wrap 语义，**不在生产 kernel 中换 global_load、降宽、改模型或增加同步掩盖它**。独立 probe、汇编、LSR 片段及失败块统计已归档。此时尚未新增 LLVM/Triton 源码改动，不把它记录成新的 PMD 缺陷。


## 固定二进制的全量频率对照

三次均使用正式 `5fde18e` 的同一个 HSACO（SHA256 `a19432c132bc6c28e7a5c5dc2e9b41ea92d8261c9f00139cedcba01ec028175f`），输入内容、所有 GPU buffer 地址和字节数完全一致。使用旧编译器缓存，不受后续 LLVM 修复影响。每次检查 67108864 个元素，0 错误；口径均为完整 dispatch、48 CU 平均，含启动和全部等待。

| GC | 完整 dispatch μs | 48 CU MMAC | 相对 2.2 GHz 耗时 |
|---|---:|---:|---:|
| 2.2 GHz | 177.095552 | 32.9739% | 基线 |
| 1.8 GHz | 180.335824 | 39.5694% | +1.83% |
| 1.75 GHz | 180.748336 | 40.5442% | +2.06% |

1.75 GHz 达到 40% 是低频利用率结果，同时耗时增加，不能当作最高频提速或 2.2 GHz 达标。1.8 GHz 的 39.57% 也不能用 wave-span 的 40.56% 代替。可复查 [完整计数与二进制一致性](gemm2-20261114/fixed-binary-frequency-comparison.json)；对应目录保存 launch、结果和编译产物 SHA。

## LLVM buffer offset wrap 修复完成

LLVM 分支 `dtk-llvm-6.3.3`，修复提交 `9021260a00b4`。修复 `splitBufferOffsets`：对于可能发生 i32 回绕的动态偏移，将不能安全拆分的立即数移回 VGPR 加法；已知低位对齐或 `nuw` 能证明安全的立即数仍保留。没有修改 kernel 的访存宽度、算法或同步，也没有修改 PMD。

原失败的 A-LDS / B 单槽原型保持相同源码，新编译器重新编译后，1394688 元素错误由 65321 降到 0。这只是原型正确性结果，尚不代表正式 kernel 提速或双 TG 驻留成功。

buffer/mubuf 相关测试共 134 项：133 通过，1 项不适用，0 失败；新增 gfx946 回归覆盖未知动态基址回绕、512 B 对齐回绕、4096 B 对齐安全立即数、`nuw` 和 store。受影响测试的预期汇编已更新；合并 store 的三组测试明确 `nuw` 前提，另有独立回绕测试保护真实问题。对更广测试中先失败的 18 项，以独立旧 llc 和原 HEAD fixture 运行其全部 RUN，确认旧版均通过，再核对新版本的必要变化。

[源码补丁、构建 SHA、保留的本地改动及测试结果](gemm2-20261114/mubuf-wrap-llvm-fix/provenance.json)。原有暂存的 CodeGenPrepare、waitcnt 和 LDS 测试改动保持原样，没有混入此次 compiler commit。正式 kernel 已完成新编译器下的独立全量复测（见下表），未以旧 cache 结果冒充新构建成绩。


## 新编译器下的 C0 先写出诊断（未替换正式 kernel）

2.2 GHz、grid48、384 输出前缀，保持 BK256、2P2C、槽位、MMAC 数与精度顺序。候选只将 P0 的 C0 buffer_store 移到等待 C1 之前，让 x 不再跨 C1 wait 保持存活。P0 实际 VGPR 55→47，消费者仍为 208/208，静态 x4 store 仍为 16 条、MMAC 仍为 256 条，scratch 0、LDS 128 KiB。

| 版本 | 前缀完整 dispatch μs | 48 CU MMAC | 错误/检查元素 |
|---|---:|---:|---:|
| 正式源码，新 LLVM | 30.249310 | 24.6443% | 0 / 8974336 |
| C0 先写出候选 | 29.510390 | 25.2613% | 0 / 8974336 |

前缀耗时降低 2.44%，尚不是完整 M4096 结论；两个全量对照已启动。候选保存在 temp，正式源码维持 `5fde18e`，没有仅凭前缀成绩替换。不同独立构建仍保留 MLS group emission 顺序差异的已知风险，不将全部差值归因于单条调度改动。


## MLS 发射顺序稳定性与收益复核

定位到 Triton `MlsOpToLLVM.cpp` 将发射分组存入 `DenseMap<Value,...>`，并直接遍历生成 MLS；Value 指针哈希使不同独立编译的组顺序不同。正式源码的新 LLVM prefix/full 构建具有同一个 TTGIR SHA（`3b3b63db...`），LLIR 却不同；差异直接表现为 A 的 0 / 16384 B 描述符组互换，已保存 diff，不能只用源码 SHA 认定机器码相同。

隔离副本只把发射分组的返回值和容器改为 `llvm::MapVector`，保留构造顺序；查表用的坐标 DenseMap 不变。原 Triton 工作树及正在运行的对照未被修改。三次独立进程 / 独立 cache 编译：LLIR、汇编和 ELF `.text` 完全一致，`.text` SHA256 为 `b44e4f86fbab4f22d41bac7e9f1bef4b958d2e84164ef01b052c5730a4ebd9d3`。三个 HSACO 只有 `.debug_line_str` 目录信息不同，不能声称整个文件相同。另一个 grid4、两 M 的 persistent 数值检查为 1394688 元素、0 错误。[源码补丁、隔离构建命令、Triton 基线/原本地改动、libtriton SHA 与逐段 hash](gemm2-20261114/mls-stable-groups/provenance.json)。

固定上述稳定构建，再比较唯一 kernel 改动（C0 先 buffer_store，再等 C1）：

| 版本 | 前缀完整 dispatch μs | 48 CU MMAC | 错误/检查元素 |
|---|---:|---:|---:|
| 正式源码，稳定 MLS | 30.155580 | 24.7209% | 0 / 8974336 |
| C0 先写出，稳定 MLS | 30.259320 | 24.6361% | 0 / 8974336 |

候选慢约 0.34%，前文不稳定构建的 2.44% 收益未得到复现，因此不合入。机器汇编确认提前写出实际发生：panel0 的四条 x4 store 位于 EMPTY12 之后、等待 C1 READY13 之前；panel1 对应 EMPTY11 / READY14，故不是源码改动被调度器撤销。当前证据支持“源码调度目标达成，但整体收益不成立”，不把寄存器减少直接当作提速。原构建的两份全量对照仍待完成，独立列结果，不覆盖这个反例。


## 对固定事务数的 credit 下界

正式 2.2 GHz `p0-scatter-repository-full` 完整计数为 1127312 个 UMC admission，全部在 dispatch 内；87 UMC cycle 的驻留窗口最多 1408 次入队，窗口长度 0.198012 μs。即使假设 tick0 即有全部 free credit、没有启动或供给空档，固定这批事务的最短跨度也至少为 `(ceil(1127312/1408)-1)*0.198012 = 158.409600 μs`，比完整 dispatch 的 40% deadline 145.988267 μs 长。

在该 deadline 内，上界是 `(floor(145.988267/0.198012)+1)*1408 = 1039104` 次入队。因此固定当前 GC/UMC 配置，要达 40%，至少需从本次 dispatch 内减少 88208 次入队（约 7.82%）；纯粹消除生产者/消费者的空档，保持事务数不变，不能达到这个 deadline。这里没有把所有 packet 当成 256 B，也没有把 admission 当成完成/credit 归还。

这是对本次固定事务数的条件下界，不是对所有算法的“不可能”结论。跨 M/N 复用、cache 合并与驻留、输出事务效率仍可能改变请求数量；每个候选必须同时记录全量 UMC 事务和耗时，不能仅凭局部 MMAC 利用率晋升。[原始计数与可复算参数](gemm2-20261114/umc-fixed-request-bound.json)。


## 新 LLVM 全量对照：C0 提前写出未通过晋升

两次都是完整 M4096 路由、3008 输出、2.2 GHz、48 CU，每次检查 67108864 元素且 0 错误；LLVM `9021260a00b4` 加已记录的原本地改动，Triton 使用原 MLS DenseMap 构建。正式 kernel 源码仍为 `5fde18e`；同一源码的新 LLIR 已存在组顺序变化，因此下表是实际构建结果，不能把差异全部归因于 kernel 写出顺序或 LLVM 修复。

| 版本 | 完整 dispatch μs | 48 CU MMAC | 有效 TFLOPS | UMC 读 | UMC 写 | 总 admission |
|---|---:|---:|---:|---:|---:|---:|
| 正式源码，新 LLVM 构建 | 173.650752 | 33.6280% | 197.87 | 603022 | 513163 | 1116185 |
| C0 先写出候选，新 LLVM 构建 | 181.016752 | 32.2596% | 189.82 | 646887 | 515653 | 1162540 |

候选慢 4.24%，多出 46355 个入队事务，其中 43865 个是读。没有单凭前缀 2.44% 的数字替换正式 kernel；正式实现保留。稳定 MLS 构建的完整对照已启动，两版都复用各自已检查前缀的整个 HSACO，输入 SHA、GPU 地址和字节数也完全一致；全量结果待完成。原始完整 dispatch perf、model 计数、UMC 窗口、config、stats 和 artifact SHA 已分别归档于对应 case 目录；本组未开启 IT/TT，不提供虚构的热点波形或 K-body 利用率。

另一个只调整配额为 56/8/224/224 的隔离诊断已开始：编译报告实际 VGPR 仍为 55/2/208/208，静态 MMAC 256 条、x4 store 16 条、LDS 128 KiB、scratch 0，未减少指令。其前缀已完成：8974336 元素 0 错误，完整 dispatch 30.482270 μs、48 CU MMAC 24.4559%，比稳定基线 30.155580 μs 慢约 1.08%；没有收益，不采用。


## 稳定 MLS 的全量结论

两次完整真实路由均检查 67108864 元素、0 错误；同一稳定 MLS 构建、LLVM 修复、2.2 GHz，并各自复用已验证前缀的完整 HSACO。两版输入 SHA、GPU 地址与字节数一致。

| 版本 | 完整 dispatch μs | 48 CU MMAC | UMC 读 | UMC 写 | 总 admission | TCC hit |
|---|---:|---:|---:|---:|---:|---:|
| 正式源码，稳定 MLS | 176.570480 | 33.0720% | 605667 | 514220 | 1119887 | 79.3247% |
| C0 提前写出，稳定 MLS | 181.689696 | 32.1401% | 649147 | 515584 | 1164731 | 78.5236% |

候选慢 2.90%，总事务增加 44844，其中读增加 43480；命令 credit stall 的 CU 均值 31293.6→33815.5，TATD credit stall 117189.6→133371.2，而 LocalMemPipeline credit stall 18955.4→18697.4。局部 LDS 等待略降，未抵消整体额外读与全局压力。TCC 命中率下降支持缓存时序受到影响，但本对照尚未将每个额外读按 tensor 归类，不能直接断言它们全是 B miss；这些阻塞计数也不能相加当作总空档。提前写出明确不晋升。[完整数据、原始 perf/config/stats](gemm2-20261114/stable-mls-full-comparison.json)。

## K64 双驻留原型：资源门槛已通过，性能待验证

保留 BM128、2P2C，每 partition 4 waves；BN128、内部 K64×4 全展开。P0 每 M 预载完整 A 到四个 K64 LDS 片段，并经 C0/C1 交接区执行 x4 scatter；P1 按 BL(k0), BR(k0), …, BL(k3), BR(k3) 搬运每 N 的 B。消费者逐段读 A/B 累加，在最后 N 的最后 K 读完后共同归还 A；P0 此时可预载下一 M，再收最后的输出。A 在 N 间留 LDS，以降低消费者寄存器需求。

A 全 K 使用 READY1/EMPTY2（4/8 waves）；BL READY3/4、EMPTY5/6，BR READY7/8、EMPTY9/10（各 4）；C0/C1 共同 READY0/13（8）与 EMPTY14/15（4）逐 panel 交接。所有首个 wait 都非 0，ID 范围仍为 0..15。BL/BR 的 ticket=4*epoch+k，ID/phase 轮换与 K 段一一对应；A 的 epoch 以完整 M 为单位。没有增加额外的生产者互锁。

LDS=32 KiB A + 8 KiB B + 16 KiB C=56 KiB。VGPR 配额 64/8/92/92 总和 256，实际 53/2/86/86，未被自动扩为 512，scratch=0。最初 SGPR98 仍可能挡住双驻留；直接 cap96 被 WDRA 以 SGPR spill 拒绝，没有绕过。发现高位 SGPR 来自 P1 的 N 地址展开；仅给后端增加 `-unroll-max-count=1 -unroll-full-max-count=1` 后，SGPR98→78，无 spill。K 的 `static_range(4)` 早已在前端展开，未受此限制。

每 SIMD 800 SGPR、粒度16：两 TG 的 8 waves 需要 ceil(78/16)*16*8=640，VGPR 静态 base64、LDS112 KiB 也均允许双驻留。两轮 grid4 数值检查为 1394688 元素、0 错误。更直接的 96TG/48CU trace 已观测 CU0 在尚无任何 wave 结束时启动了 32 个 wave、每 SIMD8个，确认实际有两个 TG 同时驻留。96TG 的前缀数值和性能仍在运行，不拿 grid4 时间冒充整卡验收。[源代码、编译命令、资源和 trace 证据](gemm2-20261114/two-resident-k64/resource-and-occupancy.json)。

另一个 32-row C panel 的缩小方案在前端因 M32 分布于 warp 维、不能直接 split 成每线程元素而未通过编译，未作为性能结果。`waves_per_eu=8` 单独设置未降低 SGPR98，未采用。


K64 原型的 96TG/48CU 前缀已完成：1536 个 BN128 输出，即前 96 个 M block；检查 **17522688 元素，0 错误**，完整 dispatch **56.113328 μs / 48 CU MMAC 26.5702%**。UMC read171556/write129743，共301299。完整 CU0 start/end 扫描确认峰值32个同时存活wave、每SIMD8个，32 starts/32 ends；CU0 16384条有效MMAC、exec0为0，逐SIMD的8-cycle区间并集等于指令数×8。32个wave的wait序列均满足首ID非0、ID在0..15、无相邻重复。[资源、数值与完整 trace 审计](gemm2-20261114/two-resident-k64-no-backend-unroll-grid96-prefix/two-tg-audit.json)。

这不是全量验收，也不能直接与早先384个BN256输出的30.156μs比较，二者工作量不同。已启动正式稳定基线的768个BN256输出/48TG对照，覆盖相同前96个M block；另单独测试K64原型将C0/C1数据都读回后再相邻写出，以检查BN128每侧128B半cache-line的合并效果。后者两轮grid4检查1394688元素0错误、SGPR78、VGPR配额仍256；其96TG同工作量前缀正在运行。尚未据此晋升新kernel。


### K64 双驻留的同工作量反例

正式版768个BN256输出/48TG与K64版1536个BN128输出/96TG都覆盖前96个M block；输入SHA、GPU地址和字节数一致，padded MMAC数相同，均检查17522688元素0错误。

| 版本 | 完整前缀dispatch μs | 48 CU MMAC | UMC读 | UMC写 | 总admission |
|---|---:|---:|---:|---:|---:|
| 正式稳定基线 | 51.026884 | 29.2188% | 153157 | 125031 | 278188 |
| K64双驻留 | 56.113328 | 26.5702% | 171556 | 129743 | 301299 |

双驻留虽然真实达成，但耗时增加9.97%，读增加12.01%、写增加3.77%、总事务增加8.31%，不晋升。TCC hit ratio从79.4491%升至85.9554%，同时外存读反而增加，不能用hit ratio代替实际事务数；K分段、N分块和并发度均已改变。BN128每侧只有128B、K64只读B列的分段，这些粒度变化可能影响请求合并/缓存时序，尚不单独归因。成对C0/C1写出的独立对照仍在跑；它只移动x的buffer_store到y读回之后，无额外同步或布局改变，已通过两轮数值检查。


成对写出对照也已完成：相同前96个M block、17522688元素0错误，58.420180 μs / 48 CU MMAC25.5210%；UMC read178205/write129155，总307360。相对K64原版56.113328 μs反而慢4.11%；写少588次，但读多6649次，不晋升，也不再为此跑全量。说明仅缩短两侧C写出的间隔没有修复此原型的额外读问题。

下一轮继续独立检验两点，尚无性能结论：正式BN256/BK256源码只禁止后端继续展开普通循环（K静态展开不变）；K64原型只把B的MLS BPS位改为0，A仍为1，T/R与指令类型保持不变。后者在另一个隔离libtriton中构建，未修改正式编译器；先做两轮数值与ISA位检查，不把它当作已验证的缓存优化。

## 最快冻结版本的完整保存（2026-11-15）

173.592048 μs / 全 dispatch 48 CU MMAC 33.639390% 的完整 3008 输出实验，67108864 元素 0 错误。算法合入提交为 `5fde18e`，但该冻结实验与合入版源码/机器码不相同。现将当时原始 `frozen-kernel.py`、`kernel.hsaco`、`kernel.amdgcn` 和 SHA256 一起保存在 `gemm2-20261114/p0-a-and-output-scatter-independent-full/`；与此前的 patch、编译元数据、结果及 perf 定位共同构成留档。源码 SHA256 为 `5915e0643689c591efda70f8ef8d8bccb36d93f3a1d7b1f5ee05811dea7cfeeb`。本次仅保存证据，正式 kernel 不变。

## 相邻 N 流水与四分区配额对照（2026-11-15，未采用）

保持 BM128/BN256/BK256、K256、2P2C、128 KiB LDS、同一稳定 MLS 工具链和 2.2 GHz。下表仅为相同前 384 输出的诊断 dispatch，均验证 8974336 元素 0 错误；不可与完整 3008 输出的 173.592048 μs / 33.639390% 直接横比。

| 改动 | P0/P1/C0/C1 配额 | 后端实际 VGPR | dispatch μs | 48 CU 全 dispatch MMAC |
|---|---|---|---:|---:|
| 正式算法控制 | 80/16/208/208 | 55/2/208/208 | 30.155580 | 24.720864% |
| 只重新分配寄存器 | 60/4/224/224 | 55/2/208/208 | 30.482270 | 24.455921% |
| 新配额 + 下一 N 的 B 预读 | 60/4/224/224 | 55/2/220/220 | 31.821790 | 23.426463% |

下一 B 预读算法：首块 B 在 N 循环前读取；当前 MMAC 后按原 FP32 顺序缩放并转为 BF16，发布第一块 64 行 C 面板，在交接第二面板之前读入下一 N 的完整 B 并归还 B EMPTY，随后发布第二面板。仍保持 B 按 N 顺序消费。首版条件更新 B 使旧 B 沿最后一轮分支继续存活，导致 WDRA 总需求超限；不放宽硬件限制。将最后一轮未使用的下一 B 显式置零后，活跃区间缩短，自动重分配为 60/4/224/224，通过 spill 检查。另一个只禁后端展开的版本也未解决首版超限，不采用。

两轮 persistent 小测（单 TG、16 输出）两版均为 299008 元素 0 错误。预读版 48 CU 的 CU0 trace 实际 16 waves、8192 条有效 MMAC、exec=0 MMAC 为 0；所有 wait ID 在 0..15、首 wait 非零且无相邻重复 wait。两项新测试输入 SHA256、GPU 地址、形状及 launch options 一致，仅 kernel source 不同。

预读版相对相同配额控制慢 4.394%，相对原配额控制慢 5.525%；只重新分配配额慢 1.083%。消费者多需 12 个 VGPR，而多给配额本身没有消除当前瓶颈。CU0 wait 发射到下一指令发射的总间隔（包含执行开销，不能相加当作墙钟时长）：P0 等 C1 第二面板的 ID14 从 148 增至 11176 cycle，P1 等 Bleft EMPTY 的 ID6 从 92664 增至 110464 cycle，消费者 Bright READY ID7 从 448 增至 2652 cycle。预读卡在两次面板交接之间，反而延迟当前输出的后半段并增加反压；并非有效隐藏了下一 B 的等待。另有分支取指等待平均每 CU 从 15206.73 增至 22166.69 cycle，需结合调度改变理解，不能单独当作唯一根因。

结论：两项均不替换正式 kernel，不为已退化的诊断继续跑完整性能测试。最佳冻结性能记录保持不变。结果、配额元数据、候选 patch、等待分析脚本、perf 定位及工具链 provenance 保存在 `gemm2-20261114/n-pipeline-quota-20261115/`。

## 用户指定的每 TG 两输出基准：面板乒乓留档（2026-11-15）

用户明确指定 48 CU、每 CU 一个 persistent TG、每 TG 两个输出 tile，共 96 输出。为保留真实 GEMM2 的 N2048/K256、所有输入与输出步长，冻结源码只将 N 遍历从 8 次截到 2 次；启动参数 TOTAL_TILES=384 仍表示前48个完整M块的归属，统计的实际输出数为96。CPU golden只检查前512列。它不是把物理 N 改成512，也不是启动两个独立 TG。所有对照输入SHA、GPU地址、BM128/BN256/BK256相同。旧长测173.592048 μs属于3008输出，不能与本表数值直接比较。

| 算法 | dispatch μs | 48 CU 全 dispatch MMAC | 数值验证 |
|---|---:|---:|---|
| 历史最快冻结算法，同口径重编 | 15.483650 | 12.036438% | 2243584元素，0错误 |
| 下一N的SB预取 | 15.208375 | 12.254301% | 同上 |
| 64行面板，保留通知但取消消费者令牌等待 | 15.035020 | 12.395594% | 同上 |
| **64行面板 + 消费者 ebarrier 乒乓** | **14.701505** | **12.676797%** | 同上 |
| 最佳面板乒乓版，同一二进制复测 | **14.701505** | **12.676797%** | 同上 |

乒乓版比历史最好算法在本次相同口径下快5.052%，比只取消等待的面板对照快2.218%。当前短测固定启动占比较大，MMAC分母仍包含完整dispatch，不用热循环口径替换。新版本每个消费者先算64行完整K256的MMAC，第一面板MMAC完成后通知另一个消费者；C0先启动，C1完成第一面板后放行C0下一轮。两个面板分别缩放、转BF16和交接。P0继续搬A并执行最终x4散写，P1继续按顺序搬左右B，仍为2P2C。

A保留原1..4协议。C0/C1 READY分别复用ID9/13并按面板0/1切换phase，原10/14用于方向性计算令牌；仍只用16个ID。C面板EMPTY保留11/12和15/0。后端自动分配80/8/212/212，实际VGPR64/2/208/208，无spill。CU0 trace含2048条有效MMAC、exec=0 MMAC为0；所有wait首ID非零、ID合法且无相邻重复。原8个N输出的单TG两M块验证也已通过，299008元素0错误。

依据用户“确认胜过同口径历史最好后保存”要求，将最佳两输出冻结源码、HSACO、汇编、复测、benchmark脚本与perf定位保存在 `gemm2-20261115/two-output-panel-pingpong/`。正式默认8-N遍历kernel未替换：其384输出面板版30.904964 μs，并未优于原384输出控制30.155580 μs。该短测收益不可泛化成长期persistent吞吐收益。长3008输出SB预取试验已按用户改用两输出的要求中止，不报告未完成的性能。

## 两输出：去除消费者 route 散读并预取下一 N 的 SB（2026-11-15）

固定 2.2 GHz、48 CU/48 persistent TG、每 TG 两个 BN256 输出，共96输出；物理 N2048/K256、原始输入和步长不变，完整 dispatch 口径。仍为 2P2C、M64面板乒乓、BK256、128 KiB LDS。

| 单项叠加 | dispatch μs | 48 CU 全 dispatch MMAC |
|---|---:|---:|
| 上一最佳，面板乒乓 | 14.701505 | 12.676797% |
| 消费者直接读取已有 SortedWeights | 14.635530 | 12.733943% |
| 上项同 HSACO 复测 | 14.635530 | 12.733943% |
| 再叠加下一 N 的 SB 提前加载 | **14.392560** | **12.948912%** |
| 组合版同 HSACO 复测 | **14.392560** | **12.948912%** |

每次验证2243584元素、0错误。组合版较上一最佳耗时下降2.10145%，距离本轮20%目标（≤11.761204μs）仍有差距。去除 route 散读单项下降0.44876%；SB预取在此基础上下降1.66014%。不将短测收益泛化为长persistent吞吐提升，正式默认8-N kernel没有替换。

路由准备已经生成 SortedWeights；新实现只把调用的权重参数从 RouteWeights 切到已有 SortedWeights，消费者按排序后的 M 连续索引读取，删除原 Sorted route 加载、CK解码、mask和按route散读权重。P0仍读取route并执行最终C散写。CPU golden仍使用原 RouteWeights，未修改数值参考或额外引入新的权重预处理，FP32计算顺序仍为 ((acc*sa)*sb)*weight。冻结调用脚本与对应源码必须配对使用，不能把SortedWeights绑定变化遗漏。

SB预取：循环外读首N；每轮保存current_sb，并在当前B消费/MMAC之前读入下一N的SB，当前两个M64面板均使用current_sb。最后一轮把无用下一值置零以缩短活跃区间。它预取的是channel scale；不把此前无收益的“在C第二面板发布前等下一B”方案作为有效优化混入。后端配额76/4/216/216、实际64/2/216/216，无spill。

CU0 trace中有效MMAC仍2048条，exec=0 MMAC为0，峰值16 waves/每SIMD4个；所有wait首ID非0、ID在0..15且无相邻重复。相比上版buffer_load_dword从420降至292（减少30.48%），s_waitcnt从436降至268；x4 scatter仍128条，pk_mul1536条、pk_bf16转换512条，缩放和转换本已成对。BL/BR READY最长等待发射到下一指令分别4/16 cycle；P0等C0/C1面板最长分别3724/2696 cycle。这些是单次wait至下一指令发射的区间，包含指令开销，不能相加当作dispatch空档；当前剩余瓶颈不能一概归因为B未到。

源码、HSACO、压缩汇编、原始launch/结果、同二进制复测、CU0审计、等待计数、工具链本地改动patch及perf定位归档于 [two-output-panel-sorted-sb-lookahead](gemm2-20261115/two-output-panel-sorted-sb-lookahead/)。提交包含本节与完整冻结证据。

BK128的先行对照未晋升：单B槽96KiB版16.311750μs/11.425384%，加入route/SA向量读与LDS转换的97KiB版16.524235μs/11.278465%，共享三个B槽的112KiB版16.837730μs/11.068475%。它们均0错误，但分段传递增加等待，腾出LDS不等于总耗时改善。把SortedWeights的x4读和较早SB预取同时叠加后，WDRA检查发现超过每SIMD寄存器预算，未放宽限制、未报告性能。继续隔离向量加载与预取的生存期。

BK128仅改为已有SortedWeights并向量化SA/weight加载（未叠加SB预取）的隔离结果：97KiB、15.399020μs/12.102588%，2243584元素0错误。比原BK128单槽快5.60%，但比新BK256最佳慢6.99%，不替换。

## 两输出：P0 读回 C0 后立即 scatter（2026-11-15）

保持上节SortedWeights、下一N的SB预取和所有MMAC/槽位/phase不变，只将P0的C0 x4 scatter从“读回C1以后”移到“读完C0、归还C0 credit以后，等待C1以前”。不让x在等待y期间继续占寄存器。2.2GHz/48CU、每TG两个输出，2243584元素0错误，dispatch **14.140490μs / 全48CU MMAC 13.179741%**；同HSACO复测相同。比上一最佳14.392560μs快1.75139%，比本轮起点14.701505μs快3.81604%，仍未达到20%目标。

后端P0实际VGPR64→48，消费者216不变；配额76/4/216/216→60/4/224/224，无spill。有效MMAC、x4 stores、metadata load与waitcnt条数全部不变；CU0硬件限制审计通过。P0等C1 READY最长2696→1548cycle，C0第二面板EMPTY最长376→604cycle，LocalMemPipeline creditStall均值318.46→251.98，global data creditStall均值58.44→0。这些仅支持此短测的交接/写出重叠改善，不能外推到长测；此前完整3008输出中类似提前写出的反例仍保留。

完整冻结代码、二进制、复测与perf定位见 [two-output-panel-sorted-sb-early-c0-scatter](gemm2-20261115/two-output-panel-sorted-sb-early-c0-scatter/)。正式默认8-N代码未替换。

负面对照：保留2P2C、由消费者自行x4散写取代P0交接写出的短测15.101450μs/12.341067%，0错误，不采用。BK128三B共享槽+向量SortedWeights/SA+P0提前scatter可在115712B LDS、实际48/2/220/220下无spill运行，15.173795μs/12.282227%，0错误，仍未胜过BK256，不晋升。单纯展开所有普通N循环的原型未通过前端编译，不报告性能。

## 逻辑 BK256 不变、消费者按 K128 读取计算（2026-11-15）

按用户要求保留完整K256的生产、槽位与phase，避免引入前述两次BK128生产的额外握手。2P2C和M64面板乒乓、SortedWeights、下一N的SB预取、P0提前scatter均保持。A完整K256在寄存器中分成两个K128操作数；B的单个32KiB槽按K方向切成两个K128视图，先读第一半并等待可用，随后发第二半DS，用第一半MMAC覆盖其等待。两半读完后只归还一次原B EMPTY；一个N仍推进一次B epoch，不因K拆分多切phase。每个面板将两次K128 MMAC累加到同一FP32 acc，乘法和BF16转换仍在完整K256累加之后。

| 对照（2.2GHz/48CU/每TG两输出） | dispatch μs | 全48CU MMAC |
|---|---:|---:|
| 上一最佳，直接K256计算 | 14.140490 | 13.179741% |
| 切K128，缺少中间编译器调度边界 | 14.481740 | 12.869172% |
| **切K128，固定首半MMAC的调度区间** | **14.110005** | **13.208216%** |
| 上项同HSACO复测 | **14.110005** | **13.208216%** |
| 同上再去掉首半显式DS等待 | 14.626430 | 12.741865% |
| 再尝试推迟A上64行读取 | 14.747460 | 12.637295% |

均2243584元素0错误。最佳版只比上一版快0.2156%，较本轮14.701505μs起点快4.0234%，未达到20%目标；严格按本次两输出范围记录，没有长测晋升。配额60/4/224/224、实际48/2/216/216、LDS128KiB，无spill。trace有效MMAC2048、exec=0为0，16个wave/每SIMD4个；barrier首ID非0、ID范围及无相邻重复审计通过。MLS、DS matrix-read、MMAC、packed乘法、BF16转换和x4写出数量均未变化，waitcnt268→300。

调度边界是编译器sched_barrier，不增加硬件同步指令。缺失时LLVM把显式wait0排到首半第一条MMAC后；补边界后首半32条MMAC能先连续发射，第二半操作数由后端按依赖插入wait，再确保wait0完成后归还槽位。不能用Python语句位置代替最终ISA的顺序验证。

等待统计脚本另修正读取barrier ID的分析问题：同一try_wait在轮询时可能多次读取同名SGPR，原脚本用后读值覆盖了首次ID，导致A READY1的部分等待误归入ID0；改为保留第一次读取值，与独立硬件审计脚本一致。此修正不改变数值、二进制、耗时或MMAC指标。

冻结代码、HSACO、汇编、复测、分析脚本、完整同口径对照和perf定位保存在 [two-output-panel-bk256-consume-k128](gemm2-20261115/two-output-panel-bk256-consume-k128/)。无收益原型未晋升。


## A 单槽分两次加载与独立元数据槽：未晋升（2026-11-15）

保留每个消费者64行输出交接区，不采用32行交接改动。独立对照把A从128×256 FP8的32KiB改为128×128的16KiB单槽，两个K128半块交替使用READY1/2、EMPTY3/4，每半块等两消费者读完才覆盖；完整A仍保存在消费者VGPR，B的完整K256槽、N相位、MMAC与输出面板不变。LDS112KiB。同口径2.2GHz、48CU、每TG两个输出，14.409850μs/全dispatch MMAC12.933375%，2243584元素0错误，硬件barrier与MMAC审计通过。慢于c8af677的14.110005μs，不晋升。

随后用腾出的空间分配独立1KiB元数据槽，P1连续x4加载SA/SortedWeights写入LDS，消费者从LDS读出，避免C0/C1各自的标量VMEM读取。覆盖前等待两侧B EMPTY，首块B READY同时发布元数据；没有复用输出交接区，也没有增加abarrier ID。LDS113KiB；14.947205μs/12.468418%，相同2243584元素0错误，CU0有效MMAC2048、exec0为0、峰值16waves及barrier审计通过。静态buffer_load_dword由73降到9，新增两条x4；但LLVM为P1 LDS写入增加ebarrier7，静态ebarrier_sync由28到29。删掉显眼的标量load不自动带来性能收益，本版不替换最佳。

临时源码、HSACO、汇编、launch、result、model-metrics和two-tg-audit分别位于/home/zhenx/temp/moe-gemm2-20261114/two-output-panel-k128-single-a-half-slot与two-output-panel-k128-single-a-half-slot-meta。工具链与c8af677冻结记录相同，没有修改LLVM或Triton；上述负面对照仅记录证据，未作为有效优化暂存或提交。跨M代际仍须独立长测，当前仅声称单M两输出数值验证通过。

上述独立槽进一步把元数据LDS写入放到首块B MLS发射之后、READY之前，以尝试覆盖VMEM回数：15.354885μs/12.137375%，2243584元素0错误，仍慢。源码与完整结果在同根目录two-output-panel-k128-single-a-half-slot-meta-overlap。该结果只支持本次调度没有收益，不表示独立LDS元数据槽本身不可行。最佳仍为c8af677。


## 首段 K128 后提前通知另一消费者（2026-11-15）

在c8af677基础上只移动C0/C1的交棒通知：首个64行面板的第一段K128 MMAC发射后，确认两段B DS读完并归还B EMPTY，随即通知另一消费者；不再等第二段K128全部发射后通知。仍为2P2C、A/BL/BR各32KiB、C0/C1各64×128 BF16的16KiB交接区，barrier ID、phase、乘法顺序和输出scatter不变。实际ISA允许通知与第二段开头少数MMAC交错，不声称两个消费者MMAC完全互斥；数据依赖与输出READY仍各自完整。

2.2GHz/48CU、每TG两个输出：**13.977145μs / 全dispatch MMAC13.333767%**；相同HSACO、相同输入和GPU地址复测完全一致，2243584元素0错误。比c8af677的14.110005μs快0.94160%，仍未达到11.761204μs目标。配额60/4/224/224、实际48/2/216/216、LDS128KiB；CU0有效MMAC2048、exec0为0、峰值16waves，barrier首ID非0、范围0..15、无相邻重复审计通过。令牌14最长等待252→4cycle，C0第二面板EMPTY最长360→4cycle；P0等C1最长1736→1584cycle。等待统计包含发射开销，不能求和当成dispatch空档。

冻结源码、HSACO、汇编、同二进制复测、数值/硬件审计、完整工具链分支/commit/本地改动索引及编译器二进制SHA在 [two-output-panel-k128-early-token](gemm2-20261115/two-output-panel-k128-early-token/)。仍保存诊断两输出版本，正式默认八N实现未覆盖。

额外负面对照均不晋升：A LDS读完后复用A区集中发布SA/weight，15.634710μs/11.920144%；再集中发布本次两输出SB，15.612870μs/11.936819%。两者2243584元素0错误，但元数据发布形成新的串行启动依赖，并需要把下一M的A预取移到当前输出排空后。将最佳版第二段B DS读提前到令牌等待前，14.205555μs/13.119375%，0错误，也没有收益。临时完整证据分别在/home/zhenx/temp/moe-gemm2-20261114/two-output-panel-k128-a-slot-metadata-overlay、two-output-panel-k128-a-overlay-all-scales、two-output-panel-k128-second-ds-before-token。当前没有新增LLVM/Triton修补。


## P0 完成缩放与 BF16 转换，消费者交出 FP32（2026-11-15）

相对c8af677改变后处理分工：C0/C1只执行MMAC并把64×128 FP32累加面板发布到LDS，P0读回后按严格原顺序((acc*SA)*SB)*SortedWeights计算、转BF16、buffer_store_dwordx4散写。保留2P2C、完整BK256生产及两段K128消费、64行交接，MMAC交棒仍沿用c8af677时机，未叠加48839e8。消费者不再执行SA/SB/weight VMEM加载、缩放或BF16转换；整个kernel的缩放工作没有减少。

FP32交接占用翻倍：A32KiB、BL32KiB、BR32KiB、独立C0 FP32交接32KiB，共128KiB；C1 FP32交接复用读完后的A32KiB。C1首面板写入发生在C0首面板MMAC交棒之后，保证两个消费者都已完成A DS；P0排空所有当前C1交接后才发下一M的A，避免跨M覆盖。代价是取消较早的下一M A预取，不能隐瞒此跨M调度变化。

2.2GHz/48CU、每TG两个输出，**13.767845μs / 全dispatch MMAC13.536468%**；相同HSACO/输入/GPU地址复测完全一致，2243584元素0错误。比48839e8快1.49744%，比c8af677快2.42495%，比本轮14.701505μs起点快6.35078%，尚未达到11.761204μs。实际分区配额116/4/196/196、实际VGPR114/2/173/173；消费者原216降到173，无spill。CU0仍2048条有效MMAC、exec0=0、16waves；barrier ID/相邻等待硬件审计通过。另按同HSACO验证每TG跨两个M、各M两个N输出（共四输出），4380672元素0错误、CU0 MMAC4096、同样硬件审计通过。该四输出仅验证跨M正确性，不混入两输出性能验收。

IT逐wave审计确认8个消费者wave的buffer_load/global_load指令合计为0，最终x4散写仍128条/CU0；FP32交接使LocalMemPipeline creditStall均值上升到1296.15，仍是需继续优化的代价。冻结源/二进制、复测、跨M脚本与验证、逐wave读取审计、编译器记录及perf路径在 [two-output-panel-k128-p0-scale-fp32](gemm2-20261115/two-output-panel-k128-p0-scale-fp32/)。正式默认八N代码未覆盖。

另测P0/P1分别完成C0/C1后处理，配额80/80/176/176、实际70/71/173/173，P1先预取下一N的B再排空当前C1，并在最终C1读回后以count2归还原A EMPTY（4个P1 wave合计8）；没有增加barrier ID。两输出13.770120μs/13.534232%，2243584元素0错误，未超过P0单独后处理版，不晋升；跨M该方案尚未验证。完整临时证据在/home/zhenx/temp/moe-gemm2-20261114/two-output-panel-k128-dual-producer-scale。


## P0 缩放叠加首段 K128 提前交棒：暂存（2026-11-15）

在d298073的P0 FP32交接/缩放版上叠加48839e8的通知位置：首个64行面板完成首段K128发射、两段B DS读完并归还B EMPTY后通知另一消费者，随后继续剩余MMAC。A/C1 LDS复用、64行交接、2P2C、全部barrier ID/count/phase与FP32缩放顺序保持不变；仍允许两消费者MMAC在交棒附近交错，不声称MMAC严格互斥。

2.2GHz/48CU、每TG两个输出，**13.571740μs / 全dispatch MMAC13.732064%**；同HSACO、相同输入和GPU地址复测完全一致，2243584元素0错误。比d298073的13.767845μs快1.42437%，比原14.701505μs快7.68469%，未达到11.761204μs目标。CU0有效MMAC2048、exec0为0、16waves，barrier ID/首等待/相邻等待限制审计通过；实际寄存器及LDS见冻结compiled.json。用户要求有效就暂存，因此源码、HSACO、汇编、复测、分析和工具链记录一并放入暂存区，尚未生成新提交。证据在 [two-output-panel-k128-p0-scale-early-token](gemm2-20261115/two-output-panel-k128-p0-scale-early-token/)。此令牌改动未另跑跨M，不能把父版本d298073跨M结果冒充本版复测。


## 验收口径更正：每 CU 跨 M 连续两个独立输出（2026-11-15）

按用户确认，后续每CU的一个persistent实例必须跨外层M块推进两次独立任务，不能只在同M下遍历两个N代替。诊断设N遍历长度1、M前缀96块、grid48，物理N2048/K256和地址跨度不变，一次dispatch共96输出；数值检查首256列。之前两N同M的13.571740μs等只作为旧口径记录，不能用于新的persistent性能确认。CPU输入SHA在以下四版中完全一致，均2190336元素0错误。所有未来调优与性能确认频率固定1.8GHz；下表2.2GHz为已有历史对照。

| 算法来源 | 2.2GHz dispatch μs | 全48CU MMAC |
|---|---:|---:|
| d7c54df，原起点 | 17.362344 | 10.734035% |
| c8af677，消费者缩放，保留下一M A预取 | 15.447250 | 12.064801% |
| 48839e8，再提前K128交棒 | 15.436785 | 12.072980% |
| d298073加提前交棒，P0缩放、FP32交接复用A槽 | 16.802240 | 11.091854% |

P0缩放候选在新口径比c8af677慢约8.77%，不能作为综合最优晋升。CU0四个SIMD在M0最后MMAC到M1首MMAC之间空档分别8164/8180/8500/8172cycle（3.71462/3.72190/3.86750/3.71826μs）；c8af677为1572/1984/2000/1984cycle。C1 FP32交接占用A LDS、P0较晚才发下一M A是该方案的重要代价。后续先验证最后C1 DS读回结束后、缩放与store之前预发下一A的安全调度，不覆盖尚未读完的交接数据。

原始launch/result/全dispatch指标、跨M间隔与可用硬件审计、相同输入核验和临时perf目录归档在 [persistent2-independent-m-comparison](gemm2-20261115/persistent2-independent-m-comparison/)。旧口径暂存的源码/二进制保留为历史证据，不据此宣称新口径收益；当前正式默认kernel没有被这些诊断N循环改动覆盖。


## 1.8GHz 新口径：消费者 x4 元数据与延后转换（2026-11-15）

以c8af677为算法起点，48CU、每CU跨两个M块、每M只计算一个N输出，单dispatch共96输出。基线17.012488μs/全dispatch MMAC13.386496%；当前16.874600μs/13.495881%，耗时降低0.81051%。相同HSACO、输入SHA与GPU地址复测16.874600μs，2190336元素torch.isclose检查0错误（rtol .02、atol .002）。尚未达到20%目标。

保留2P2C及下一M的A预取。P0负责A MLS预取、读回BF16面板与x4输出scatter；P1负责Bleft/Bright MLS。消费者各算128列，分两个64行面板、每面板按两段K128累加；仍由消费者完成严格((acc*SA)*SB)*SortedWeights缩放及BF16转换。SA/SortedWeights改用连续x4加载，通过两个波内寄存器/通道交换转换布局，延至首面板完整K256 MMAC发射后执行；空的tied-register identity只防止布局转换被MLIR合并为共享内存fallback，不发射ISA。LDS仍128KiB，实际VGPR36/2/204/204；CU0有效MMAC2048、exec0=0、峰值16waves与barrier审计通过。

提前转换的单独对照17.069200μs；P0集中缩放并在旧尾部DS读回后提前加载下一A为17.829252μs，均未超过基线，不晋升。当前仅暂存冻结诊断代码和证据，正式默认八N kernel未覆盖。基线与有效优化分别在 [persistent2-independent-m-c8-gc1800](gemm2-20261115/persistent2-independent-m-c8-gc1800/) 和 [persistent2-independent-m-c-vector4-late-convert-gc1800](gemm2-20261115/persistent2-independent-m-c-vector4-late-convert-gc1800/)，包含工具链分支/commit/本地改动索引、二进制哈希、原始perf位置、数值/硬件审计及同二进制复测。

本版另作三组1.8GHz单因素对照，均为跨两个M、96输出，2190336元素0错误：提前K128交棒17.023608μs；P0输出布局每线程连续四行16.900176μs；B scale加载移到A READY等待之前17.035284μs。均未超过16.874600μs，不晋升。原始诊断证据在/home/zhenx/temp/moe-gemm2-20261114/persistent2-independent-m-c-vector4-late-convert-{early-token,p0-row4,sb-early}-gc1800；提前交棒版另通过barrier与有效MMAC审计。

面板顺序另作对照：两面板完整MMAC算完后再交棒及缩放，18.248476μs；保持原首面板交棒位置、仅把后处理放到两面板MMAC之后，17.531236μs。两者2190336元素0错误、LDS128KiB；前者实际VGPR36/2/208/204，硬件barrier审计通过。MMAC间隙减少不代表dispatch更快，延后输出交接和C1启动增加了流水代价，均不晋升。单面板MMAC期间暂时提升优先级为2、后处理恢复0，17.095888μs，0错误，同样不晋升。临时完整证据分别在同根目录persistent2-independent-m-two-panels-before-scale-gc1800、persistent2-independent-m-two-panels-original-token-gc1800、persistent2-independent-m-c-vector4-mmac-prio2-gc1800。

P0当前路由准备前移到下一A槽等待之前为16.991360μs，仍慢于最佳；C1 FP32交接改为复用Bright32KiB、P0读完最后面板才归还BR EMPTY8，使A与输出交接独立、恢复下一M A提前预取，为17.262688μs。均2190336元素0错误、128KiB LDS，后者实际VGPR83/2/164/164。此槽位对照未胜出，不替换消费者缩放的最优方案。证据在同根目录persistent2-independent-m-c-vector4-route-before-next-a-gc1800与persistent2-independent-m-p0-scale-c1-alias-br-gc1800。

x2元数据单次波内交换对照17.198748μs，2190336元素0错误，未胜出。实际ISA的静态x4元数据16条变为32条x2，v_permlane32_swap_b32由32条变为0，v_permlane16_swap_b32仍32条，没有ds_bpermute；减少布局交换未抵消增加的VMEM命令代价。临时完整证据在persistent2-independent-m-c-vector2-one-swap-gc1800。当前仍保留16.874600μs、全48CU dispatch MMAC13.495881%的x4延后转换版；本轮无额外有效优化，负面对照仅保存指标和诊断路径，不晋升实现。


## 消费者循环外预取与首波发射顺序（2026-11-15）

按用户要求验证SA/SortedWeights/SB在消费者循环外预取，仍驻留VGPR；后续M的元数据在前一M末面板发布前预取。1.8GHz、48CU、跨两个独立M输出：单纯外提17.668568μs；首批生产者临时优先级2、发完恢复0为17.136476μs。两者2190336元素0错误，但trace显示即使临时优先级，部分消费者buffer_load仍先于P0/P1 MLS，不能声称满足首波顺序。

严格门控复用原初始MMAC令牌10：count改8，取消共同初始化时的预先到达；P0/P1各4waves在首批MLS发射后到达，消费者循环外先等令牌10 phase0。后续C1每wave arrive10 count2，保持后续令牌代际；14仍count4，不新增ID。已对CU0全部4SIMD逐wave审计：8个生产者wave的首批MLS都先于8个消费者wave的首条元数据buffer_load；消费者不等待MLS回数完成，只等待首波指令发射通知。barrier首ID非0、无相邻重复、有效MMAC2048/exec0=0、16waves及数值2190336元素0错误均通过。耗时18.281836μs，未超过16.874600μs，不晋升。

临时完整证据在/home/zhenx/temp/moe-gemm2-20261114/persistent2-independent-m-scale-prologue-{prefetch,producer-first,issue-gate}-gc1800；严格门控新增审计脚本audit_scale_issue_gate.py、结果scale-issue-gate-audit.json。上述仅为用户指定调度对照，当前最优实现与暂存的最优二进制保持不变。


## BK128 与生产者 LDS 元数据：方向验证（2026-11-15）

全部沿用1.8GHz、48CU、每CU跨两个独立M输出，一次dispatch共96输出，2190336元素torch.isclose（rtol .02/atol .002）0错误。2P2C保持不变。以下均为未提交诊断实现，不把HEAD或历史commit当作新候选源码标识；逐版launch保存源码SHA。历史最优仍为消费者VGPR元数据的16.874600μs/13.495881%。没有修改正式默认kernel或编译器。

| 算法特点 | LDS KiB | dispatch μs | 全48CU MMAC |
|---|---:|---:|---:|
| BK256，仅交接区N128→N64，仍消费者VMEM |112|17.480640|13.027990%|
| 同交接区，P0加载SA/weight、P1加载SB并写LDS |114|17.160940|13.270695%|
| BK128，A单K128槽，B左右各单槽，完整N128交接；scale/weight/route/expert放LDS |83|20.146104|11.304300%|
| BK128，A恢复整K256槽及下一M整A预取，其他同上 |99|17.287152|13.173807%|
| 再于首个Bleft MLS之后同时预取左右SB |99|17.314396|见JSON|
| A使用READY1/EMPTY3，释放ID2/4给第二Bleft K128槽 |115|17.247676|13.203959%|
| Bleft双槽、每完整输出才交棒 |115|17.716384|见JSON|
| Bleft双槽、元数据按M双槽并在下一A等待前预取 |117|17.958800|见JSON|
| Bright双槽，K1先发Bright，修正跨M相邻EMPTY6等待 |115|17.062528|见JSON|
| Bleft双槽，SA/weight LDS按消费者行位序排列 |115|17.112012|见JSON|

元数据3KiB划分为SA512B、weight512B、route512B、expert512B、左右SB各512B。P0在A READY之前完成前四项LDS写入；消费者读完SA/weight/A并wait0后归还A EMPTY，P0也在下一M覆盖前读完当前route/expert。P1对每N的首K半块发布SB，B READY包含该LDS写入，消费者在B EMPTY之前已读回SB到寄存器。后续K半块沿用该N的SB。不存在A/B矩阵和元数据的物理别名。

A单K128槽版每半块读完立即归还EMPTY，不能读完两个半块才归还；跨M也不能在当前输出排空前阻塞预发下一M的第二A半块，否则会与输出交接形成环等待。A整K256槽版不需要该分段尾首协议，恢复原整A提前预取，83KiB→99KiB虽然占用增加，耗时大幅下降。说明空闲LDS和低寄存器数本身不是优化目标。

双Bleft槽利用A的两对ID合并后释放的ID：A READY1/EMPTY3；Bleft两槽READY5/2、EMPTY6/4；Bright READY7/EMPTY8；输出和令牌9..15/0不变。P0在下一次EMPTY3前先观察由自己发布的当前READY1完成，避免相邻重复等待ID，同时不能观察尚未发布的未来代际。Bright双槽初版虽数值0错误，但硬件审计发现跨M的EMPTY6相邻重复，剔除；修正版在下一EMPTY6前先观察P1自己刚发布的当前K1 READY5，不新增ID。不能以PMD数值通过代替硬件等待约束审计。

布局对照将一维LDS物理offset位映射为逻辑行位[4,8,1,2,16,32,64]，使消费者寄存器中的行4/8相邻值成为四元素连续物理地址。SA/weight静态ds_read2_b32合计32条降为16条ds_read_b128，无消费者permlane或VMEM；逻辑值、FP32缩放顺序、barrier及128行计算分工保持一致。普通布局Bleft双槽的LocalMemPipeline.creditStall均值1594，高于历史最优418.15，表明把VMEM搬入LDS后还需处理局部命令压力。

阶段分析包含短K循环、MMAC后处理、跨M交接：CU0 SIMD0中，历史最优C0/C1各自M0最后MMAC到M1首MMAC间隔2784/3172cycle；Bleft双槽普通布局为3480/3412cycle。P0从M0最后store到M1首store间隔1300→2844cycle。单看MMAC发射更紧或删掉消费者load不足以证明整体收益。该间隔包含后处理、准备及调度，不将跨wave时间相加当成dispatch。

原始指标、输入SHA、源码SHA、可用硬件/消费者VMEM审计与三段时间分析在 [bk128-producer-metadata-study](gemm2-20261115/bk128-producer-metadata-study/)。无历史最优收益的原型只记录证据，不暂存其实现。


### BK128 最优独立暂存（2026-11-15）

按用户最新保存规则，BK128 当前最好17.062528 μs已单独冻结至 [BK128 最优源码与证据](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md)。后续仅超过该BK128最好更新暂存；超过全历史16.874600 μs并复测才生成新commit。本次没有将较慢方案替换为正式默认实现。


### BK128 超过历史最好：独立元数据 READY（2026-11-15）

本节与 [对应冻结源码、HSACO和复测证据](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md) 一并提交，具体提交可用本节git blame或该目录git log定位。按1.8GHz、48CU、每persistent TG两个独立M输出的同口径，新结果 **16.632184 μs / 全dispatch MMAC 13.692585%**；同HSACO、同输入SHA复测完全一致，两次2190336元素均0超差。相对全历史16.874600 μs降低1.4366%，相对BK128旧最好17.062528 μs降低2.5222%。20%目标尚未达成。

| BK128算法变化 | dispatch μs | 结论 |
|---|---:|---|
| Bright双槽、元数据与A共用READY，消费者缩放 |17.062528|上一BK128最好|
| P0缩放FP32，N64交接，普通写出布局 |22.017044|写出指令翻倍，消费者输出credit等待严重|
| 上版改为每线程8个连续BF16的写出布局 |18.840616|修正向量宽度仍不及旧最好|
| A直接分四个子块DS读取 |17.314952|数值通过，未获益|
| P0保留route/expert VGPR，省去回读 |17.275476|布局转换仍引入额外LDS scratch|
| route直接按P0写出布局加载 |17.241004|取消scratch仍无整体收益|
| A/MA独立READY、MA按M双缓冲，普通MA布局 |17.166500|脱离A等待，但后处理LDS读仍较多|
| 独立READY加MA向量布局，MMAC后才读SA/weight |**16.632184**|新全历史最好；已同二进制复测|

关键收益是A不再串行等待SA/weight/route/expert完成；消费者先读A并释放credit，完成当前MMAC后读SA/weight。MA按M双缓冲，READY也按两个槽分别用12/0，phase按每槽代际推进。初版只双缓冲内存、共用一个READY ID时，在生产者提前发布下一M后丢失旧代际，已修正，失败版不计入性能。MA物理行位序让SA/weight静态32条ds_read2_b32改为16条ds_read_b128。实际VGPR39/9/146/146，LDS117KiB，无spill。

释放元数据READY ID的方式是每消费者只用一个输出EMPTY ID（11或15）；第二panel前观察自己已发布的前一READY，再等待EMPTY，确保无相邻同ID等待。没有突破16个abarrier或首wait不能0的限制。全16wave、MMAC计数、exec、首wait/相邻等待和数值均审计通过；消费者无全局load或permlane。

CU0 SIMD0的新C0/C1跨M MMAC间隔4252/4316cycle，仍有优化空间；不能据总耗时改善声称每个局部间隙均已缩短。该提交只保存固定N遍历截为1的两M诊断实现，正式默认kernel尚未替换，完整N遍历未由此次测试证明。编译器及PMD与原最好相同，分支commit/本地差异在冻结工具链证明内。


### BK128 继续提升：scale LDS 读取与后半K重叠（2026-11-15）

上一版16.632184 μs已随源码/报告保存为commit `1f2572a`。本版仍保留独立A/MA READY及下一M整A预取，只把消费者SA/weight的LDS读取从完整K结束后，提前至第一个K128 MMAC并交棒之后，不提前到首波生产者之前。读取等待被下一个K128的B读取/计算覆盖，数值缩放顺序不变。

结果 **16.374200 μs / 全48CU dispatch MMAC 13.908319%**。同HSACO、同输入SHA复测完全一致，2190336元素0超差，16wave、MMAC数、exec和等待约束审计通过。实际VGPR39/9/174/174，LDS117KiB。相比16.632184 μs减少1.5511%，相对原全历史16.874600 μs累计减少2.9654%；20%目标尚未达成。对应源码/二进制/结果已更新 [BK128最好归档](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md)，正式默认kernel尚未替换。


### Store 发射、mask 常量裁剪与编译器顺序排查（2026-11-15）

0054799的P0 SIMD0第一次等C0 READY为2264cycle；写出期间还存在1272/1448cycle的结果等待。密集段每wave的x4 store通常隔64cycle，合并四SIMD通常13~17cycle。CU0 GM cmd/addr/data credit为422/32/0，TA对TCP反压576cycle；这些不是store专属统计，不能将所有空档归给写出队列。

N2048能被BN256整除，K256能被BK128整除，列/K边界mask已经消失；CK topk8仍要路由解码。填充行15360/48128，行mask不能整体删除。本地expert配置常量裁剪、uniform expert、地址提前、优先排空C0、改变输出swizzle均未优于16.374200μs，没有替换最好实现。详细结果与算法标识见 [store-issue-and-credit-order](gemm2-20261115/store-issue-and-credit-order/README.md)。

条件跳过完全无效64行panel暴露LLVM MachineSink缺陷：原LLIR中LDS读在EMPTY归还之前，汇编及IT却在归还之后读，187678/2190336元素错误。同二进制复现，输入有效前缀断言通过；加入不生成硬件指令的memory-clobber对照后0错误，证明应修编译器内存排序，不能当作PMD旧缺陷或加硬件同步掩盖。最小LLVM输入/汇编、正确对照和待验证补丁均已记录；LLVM分支dtk-llvm-6.3.3/9021260a00b4及既有本地改动保持，第一版SOPInstructions.td内存属性修复因引入buffer_wbinvl1_vol已撤回；已收窄为SIInstrInfo.cpp的MachineSink合法性限制及回归用例，LLVM工作区diff尚未提交。新增最小回归与两项既有barrier测试3/3通过；不新增硬件同步或缓存失效指令。

最终修复后，同口径原最好control为**16.374200μs / 13.908319%**，与原编译器结果一致。条件跳过整组无效行版为16.537108μs / 13.771308%，慢约0.995%，不替换最好kernel。两版2190336元素零超差，16waves、2048有效MMAC、exec0=0、barrier等待约束审计通过，ISA均无buffer_wbinvl1。补丁、三项测试日志、编译器新SHA及已有LLVM本地改动、逐版指标已记录在上述证据目录；Triton/PMD保持原最好工具链版本。此次修复的是LLVM正确性，不计为性能提升，也未生成新性能胜出版commit。


### BK128 第二K相位按面板交棒：新最好16.134008μs（2026-11-15）

与源码和完整证据一并提交，具体commit由本节git blame或[BK128最好归档](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md)的git log定位。1.8GHz/48CU/每TG两个独立M输出（共96）同口径，**16.134008μs / 全dispatch MMAC14.115377%**，比0054799的16.374200μs降低**1.4669%**。同HSACO、源码SHA、输入SHA及GPU地址复测16.134008μs，2190336元素零超差。20%目标仍未达成。

2P2C、117KiB槽位、A/B发射顺序及精确FP32乘法顺序均保持。K0仍两面板一起交棒；K1低面板完成后先交棒、缩放发布，再等令牌算高面板并交棒。这样整数后处理、BF16转换和发布能与另一消费者的MMAC重叠。令牌变为每M三次，phase必须按(3*Mepoch+subphase)&1；IT读取SGPR验证8消费者wave跨两个M为0/1/0/1/0/1，A/B代际不变。barrier ID/首wait/相邻等待、16waves、2048有效MMAC/CU、exec0=0及消费者0VMEM/0permlane均通过。实际VGPR39/9/174/174，无spill。

CU0 SIMD0 C0/C1跨M计算空档3340/3420→1936/2308cycle，最后MMAC到最后READY2588/2664→1140/1236cycle，P0跨M store空档2436→2208cycle。首次MMAC分别晚416/384cycle，局部启动并非全面改善，但尾部重叠使dispatch更短。不同wave时长不相加当作dispatch收益。

| 对照算法 | dispatch μs | 全48CU MMAC | 处理 |
|---|---:|---:|---|
| 父版0054799，原调度+修复后LLVM control |16.374200|13.908319%|同原编译器结果|
| B READY独立，首两块B后加载SB，共同MA/SB READY |16.845688|13.519044%|不晋升|
| B READY独立，四块B后加载SB |17.378892|13.104265%|不晋升|
| P1分担C1散写，先预发下一M首对B；P0仅写C0 |17.441164|见JSON|不晋升|
| K1低/高面板分别交棒，三次令牌/M |**16.134008**|**14.115377%**|新最好，同二进制复测通过|

负面对照数值均零超差并通过等待约束审计，源码未晋升；测量与源码SHA见[k1-panel-handoff-study](gemm2-20261115/k1-panel-handoff-study/comparison.json)。P1散写首版的新增自观察EMPTY取错phase导致停滞，已终止；修正版观察panel^1后才得到表中结果，停滞版不参与性能排名。SB拆分两版的首次MMAC反而推迟，不能将B READY提前等同于实际计算提前。

新最好包含LLVM MachineSink修复，原算法control确认该修复不改变原最好耗时；LLVM完整tracked本地补丁与新增回归均随最好归档保存，Triton/PMD二进制SHA保持原版本。正式默认kernel尚未替换，完整N遍历未由此次两M诊断证明。


### K0 同样按面板交棒：继续提升至15.887144μs（2026-11-15）

三次交棒版已保存为commit `1571b2f`。本版K0也在低面板MMAC后交棒，C1不必等C0两个面板都发完；K1沿用低面板完成后交棒并做后处理。每M四次token phase为0/1/0/1，下M重复。K0两次同ID token wait之间观察自己已完成的B EMPTY（phase^1），满足硬件不得相邻同ID等待的约束；未观察未发布的未来代际。初稿发现相邻token wait未分隔，已终止且不计性能，表中仅为修正版。

同口径**15.887144μs / 全48CU dispatch MMAC14.334710%**；同HSACO、源码SHA、输入SHA及地址复测完全一致，2190336元素零超差，硬件等待约束、每消费者四次交棒/M、A/B代际、16waves、2048有效MMAC/CU、exec0=0、消费者0VMEM/0permlane均通过。比1571b2f再降1.5301%，比0054799累计降2.9745%；20%目标尚未达到。2P2C、117KiB LDS、39/9/174/174实际VGPR和精确FP32乘法顺序不变，工具链与上一版相同。

CU0 SIMD0 C0/C1跨M计算空档1752/1852cycle（上一版1936/2308）；最后MMAC到最终READY1100/1196cycle（上一版1140/1236）。相比0054799，首次MMAC提前24/344cycle。P0跨M store间隔2312cycle，较上一版2208略增，但全dispatch仍更短。源码、perf索引、工具链补丁和同二进制复测已更新[BK128最好归档](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md)，结果随代码一并提交；正式默认kernel及完整N遍历尚未覆盖。


657c18f后的单因素对照（同1.8GHz/48CU/两独立M）：仅P0优先级1为16.086748μs，仅P1优先级1为16.034484μs；将P1下一M的expert索引load移至当前B末块READY5等待之前为16.195724μs。均2190336元素零超差，均未超过15.887144μs，未暂存其实现；仅记录[后续对照指标](gemm2-20261115/four-handoff-followups/comparison.json)。当前最好维持657c18f。


### K1 按32行计算、64行发布：15.783728μs（2026-11-15）

在commit `657c18f` 上保持BM128、2P2C与117KiB槽位不变，K0两个64行面板，K1四个32行面板分别MMAC/交棒/缩放。相邻两块分别写原co的前后32行，第一块写前等EMPTY、第二块写完wait0再发布一次64行READY；P0写出粒度与交接次数保持原方案。每M六次token phase交替，K1中间通过观察自身已发布READY分隔相邻同ID token wait。全8消费者wave的真实SGPR phase、A/B代际与等待约束审计通过。

同口径**15.783728μs / 全48CU dispatch MMAC14.428632%**，比657c18f的15.887144μs降低**0.6509%**。同HSACO、源码/输入SHA、GPU地址复测完全一致，2190336元素零超差，16waves、2048有效MMAC/CU、exec0=0、消费者0VMEM/0permlane通过。实际VGPR39/9/174/174。20%目标尚未达成。

单独32行逐块发布虽然co缩至每消费者8KiB、LDS101KiB、VGPR39/9/170/170，但16.215740μs，慢于父版，不晋升。整块下一M A LDS预读（最后MMAC后、当前scale前发出）16.756172μs亦无收益。对照指标、数值与源码SHA见[m32-panel-study](gemm2-20261115/m32-panel-study/comparison.json)。失败性能原型未暂存其实现。

新最好CU0 SIMD0 C0/C1跨M空档1412/1596cycle（父版1752/1852）；最后MMAC到最终READY612/748cycle（父版1100/1196），首次MMAC晚104/436cycle，尾部收益抵消启动代价。源码、二进制、工具链证明、perf索引和复测随本报告一并提交到[BK128最好归档](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md)。正式默认kernel与完整N遍历未由此次诊断覆盖。


### 保留数据依赖、取消额外计算token：15.387300μs（2026-11-15）

父版`c11d0d8`保留全部A/B/MA/输出READY与EMPTY，只移除C0/C1计算之间额外的AB10/14交棒；K0自观察EMPTY随其删除，输出前一READY自观察延至真正复用输出EMPTY之前。2P2C、K0 M64/K1 M32消费、每两块合并发布64行、117KiB槽位、39/9/174/174寄存器与原FP32顺序均保持。两个消费者的计算由硬件调度，未弱化数据依赖。

同口径**15.387300μs / 全48CU dispatch MMAC14.800361%**，比c11d0d8降低**2.5116%**。同HSACO、源码/输入SHA、GPU地址复测完全一致，2190336元素零超差。16waves、2048有效MMAC/CU、exec0=0、8cycle忙区间并集无重复计时、A/B真实代际、无额外计算token、abarrier ID/首wait/相邻等待及消费者0VMEM/0permlane审计通过。20%目标尚未达成。证据与源码随本报告一并提交到[BK128最好归档](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md)。

对照：A分两段K128读并延迟EMPTY3至第二段完成为16.374756μs，首次计算虽提前但M0计算跨度变长，不晋升；使用EB14/15整组count8交棒为15.886588μs；使用EB10..13两组（各两个SIMD、count4）为15.694212μs，均数值零超差且参与wave/计数/ID已审计，但都慢于无计算token版。结果见[compute-handoff-study](gemm2-20261115/compute-handoff-study/comparison.json)。失败性能原型未暂存源码。

首版每SIMD一对EB8..15发生死锁，检查IR/ASM发现编译器为输出布局转换自动使用EB8/9（count4），与手写count2冲突，已终止，不排名。后端ConvertWarpSpecializeToLLVM.cpp保留EB0..5，各分区占用6..9；有效对照只使用10..15。两组EB按逻辑wave索引分组，IT确认对应物理SIMD集合为0/2与1/3，不能把逻辑wave低位直接当物理SIMD编号。ID审计及失败原因见同目录，并非PMD缺陷。

当前CU0 SIMD0跨M计算gap1600/1828cycle，较父版1412/1596增加；但M0 C0计算窗口5144→4572cycle，P0跨M store间隔2164cycle，全dispatch仍缩短。局部gap不直接代表总性能。正式默认kernel与完整N遍历未由此次诊断覆盖。


f0fa448后续控制对照仍用1.8GHz/48CU/每TG两个独立M，均2190336元素零超差：32行逐块发布的单Bleft槽15.767604μs/101KiB；双Bleft槽15.815420μs/117KiB，READY10/EMPTY14真实代际及16ID约束已核对，无槽位闲置，但多一槽未获益。P0两边联合回读15.626380μs，C0先归还再合并store15.459024μs，均未胜出。仅改配额为64/16/208/224仍15.387300μs；消费者各192或176也完全一致，实际VGPR不变且无scratch spill，不晋升这些配额。结果与源码SHA在[no-token-followups](gemm2-20261115/no-token-followups/comparison.json)，无效性能实现未暂存；当前最好保持f0fa448。


### A MLS 发射窗口优先级2：15.327252μs（2026-11-15）

在f0fa448上仅在P0 A EMPTY/MLS发射窗口设置priority2，A READY arrive后恢复0，metadata和输出不提升，P1保持0。结果**15.327252μs / 全48CU dispatch MMAC14.858345%**，比f0fa448降低**0.3902%**。同HSACO、源码/输入SHA、GPU地址复测一致，2190336元素零超差。2P2C、117KiB LDS、39/9/174/174寄存器、原FP32乘法顺序不变；A/B代际、无计算token、16waves、2048有效MMAC/CU、exec0=0、忙8cycle区间无重复计时、abarrier等待约束和消费者0VMEM/0permlane均通过。20%目标尚未达成。

IT优先级窗口审计确认4个P0wave每个2个窗口，每窗8条MLS；高优先级窗口内没有metadata/global output load/store，并恢复priority0。C0首次MMAC早316cycle，C1晚256cycle，不能以小幅整体收益声称每个局部都改善。A/B分别2/1的发射窗口对照15.375624μs，不及A-only；四块B独立READY之后再加载SB、与MA汇合发布的对照16.375312μs，未胜出。证据见[a-issue-priority-study](gemm2-20261115/a-issue-priority-study/comparison.json)。新最好源码、二进制、工具链、perf索引与复测随报告一并保存到[BK128最好归档](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md)。正式默认kernel与完整N遍历仍未由此次诊断覆盖。

SB生命周期补充：旧“首对B后发下一M SB”的负面对照仅有数值/等待约束通过，未能保证下一M写MB前C1已经读完旧SB，不得晋升。四块B之后发布则有生命周期依据：下一M BR1必须先取得上一M C1 K1的EMPTY4，而C1在归还该credit前的wait0已捕获旧SB；Bleft同理。当前最好仍保留原KH0 B READY包含SB完成的协议，不存在该复用风险。


### 非Gluon constexpr mask 对照复核（2026-11-15）

参考非Gluon `moe_op.py::decode_sorted_token_id` 的 `ck_topk==1` 分支，以及 `block_n_diviable/block_k_diviable` 的编译期边界裁剪。当前最好264d7e7的N2048/BN256、K256/BK128已无列/K边界mask；CK/topk8的decoded route非负检查也已被编译器消除。topk1可省route高8位解码，但当前topk8不满足条件，不能作为当前性能收益。padding行和无效expert由运行时路由决定，仍需保护。

单因素源码对照将 `route>=0 && route<valid_routes` 改为unsigned范围比较（valid_routes非负int32），保留expert检查并增加topk1编译期解码分支。实际ISA均1503条，仅8条signed比较改成unsigned，其他指令完全相同。1.8GHz/48CU/每TG两个独立M输出下仍为**15.327252μs / 全dispatch MMAC14.858345%**，2190336元素零超差，无性能收益；不暂存候选实现，最好版本仍264d7e7。完整N遍历和topk1未由此次性能对照验证。证据见[constexpr-mask-review](gemm2-20261115/constexpr-mask-review/comparison.json)。


### 跨M预读与输出LDS布局对照（2026-11-15）

在264d7e7上按同1.8GHz/48CU/每TG两个独立M输出口径继续检查：下一M A提前读15.916056μs；消除零占位的prologue版15.761488μs；A/B0/SB一起预读16.366972μs；下一M K0低面板移到旧M最后缩放之前17.142592μs。均2190336元素零超差，后者实际消费者VGPR214/214。局部轮间gap虽缩短，但当前计算窗口和发布延后，整体未获益。

输出Co原生vector8布局18.406380μs，行XOR16.487624μs，补列XOR16.040600μs；消费者宽写导致P0回读变零散。普通Co Swizzled(32,1,4)把平均每CU bank-conflict计数1280降到768，仍15.412320μs。仅删未用AB初始化15.467920μs，仅缩放转换窗口priority1为15.368396μs。没有胜出版，不暂存这些候选实现；当前最好仍**264d7e7 / 15.327252μs / 全dispatch MMAC14.858345%**，20%目标未达成。算法、完整指标、数值与已执行审计见[跨M预读与Co布局对照](gemm2-20261115/cross-m-prefetch-and-co-layout/README.md)。工具链无新改动，正式默认kernel及完整N遍历未由这组诊断覆盖。


### A与首块B共用DS等待：15.282772μs（2026-11-15）

在264d7e7上，把A LDS读取后的wait0和EMPTY3延后到首块B/SB读取之后；一次wait0完成A/B/SB，随后分别归还A EMPTY3和首B EMPTY。MA元数据时机、2P2C、117KiB槽位、39/9/174/174寄存器、精确FP32顺序和x4输出均保持。真实IT专门审计8消费者wave各两个M，24条matrix DS读及SB read2在一次wait0前，A EMPTY在wait0后，未提前释放槽位。

1.8GHz/48CU/每TG两个独立M输出，**15.282772μs / 全48CU dispatch MMAC14.901590%**，比264d7e7降低0.2902%。同HSACO、源码SHA、输入SHA、GPU地址复测完全一致，2190336元素零超差。A/B代际、16waves、2048有效MMAC/CU、exec0=0、忙8cycle并集、AB等待限制、消费者0VMEM/0permlane和A-only priority2窗口通过。新最好源码、HSACO、perf索引、审计和工具链证明更新到[BK128最好归档](gemm2-20261115/persistent2-independent-m-bk128-best-gc1800/README.md)。20%目标未达到；正式默认kernel和完整N遍历仍未覆盖。

负面对照：P0右半输出改为C+BN/2独立SRD，offset/mask复用确实省去8条v_or和8条v_cndmask，仍为x4，但16.247432μs；把MA的SA/weight也并入首次DS等待为15.574672μs。均零超差，没有晋升其实现。


8836708后续对照仍为1.8GHz/48CU/每TG两个独立M输出，均2190336元素零超差并通过MMAC数量/16wave及A/B代际和等待约束审计：C1将B1读取放在K0低/高面板之间为15.561884μs；仅后续M提前、首M保持原流程为15.887144μs；P1先把两个SB半块buffer load到寄存器、LDS写仍在各自B EMPTY保护之后为16.034484μs；P1 expert索引由global load换成单元素buffer load为15.346156μs。都未胜过15.282772μs，不晋升源码。CU0 SIMD0 C1提前B1等待在M0为328cycle、M1为8cycle，父版对应4/4；不能把B1预读收益外推到冷启动。数据见[combined-ds-followups](gemm2-20261115/combined-ds-followups/comparison.json)。


### BN256 tuning frozen; independent BN512/BK64 experiment (2026-11-15)

Per user instruction, BN256 optimization stops at8836708 /15.282772us. The already-running wide-A study finished before switching: isolated lowering correction passes all numerical/phase/wait/MMAC/consumer-VMEM audits, but wider A is slower (15.513512/15.673084/15.661408us for producer warps2x2/4x1/1x4). Original-path native instructions and15.282772us remain identical. No wide-A implementation or diagnostic compiler was promoted; see[wide-mls-study](gemm2-20261115/wide-mls-study/README.md).

Further work uses the independent `gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn512xbk64.py` file. First full256-column consumer attempt exceeds the512-VGPR total budget: raw demand40/9/256/256 with spills. Splitting each consumer's256 columns into two128-column phases reuses A and metadata while reducing actual demand to40/9/187/187, no spills. The first single-Bleft-slot prototype passed numerical comparison but failed the hardware rule against adjacent same-ID barrier waits after expanding K to four chunks; it is rejected irrespective of PMD timing. The current two-Bleft/two-Bright-slot variant uses85KiB LDS, AB5/10 READY and6/14 EMPTY for Bleft; AB7/2 and8/4 for Bright. Both cycle phase0,0,1,1 for K0..3 and reset correctly between N phases. Numerical/performance and instruction-trace audits are in progress; no performance win or20% target claim yet.


BN512/BK64 的独立文件已完成首个有效基线：**22.131580μs / 全48CU dispatch MMAC20.580329%**，每TG两个128×512独立M输出，4380672有效元素零超差。清理后的仓库文件和带旧helper的原型分别运行，结果一致；文件字节与归档一致。85KiB LDS、40/9/187/187 VGPR、无spill。真实IT的16waves、4096有效MMAC/CU、exec0=0、忙8cycle并集、首wait非0/无相邻同ID、A/B代际、消费者0VMEM及P0 A priority2窗口均通过。源码、二进制、工具链来源、命令和perf索引见[BN512/BK64基线](gemm2-20261115/bn512-bk64-baseline-gc1800/README.md)。这里只验证前512列，完整N2048仍未覆盖；宽度加倍后工作量也加倍，不把22.13和原BN256的15.28直接比较或声称20%目标达成。

后续仅在独立BN512文件做两个对照：Co改成M64、两个M32合并发布为23.009504μs/19.795090%；两个N128阶段静态展开为23.033968μs（完整MMAC指标见归档）。均数值/代际/等待/无spill检查通过，但不及22.131580μs；保留M32发布、动态两N阶段的基线文件，不晋升这两个候选实现。CU0 C1跨M空档在Co64对照约2176~2216cycle，基线1852~1952cycle，发布次数减少并未改善实际衔接。[对照数据](gemm2-20261115/bn512-bk64-study/comparison.json)。BN256继续冻结于8836708。
