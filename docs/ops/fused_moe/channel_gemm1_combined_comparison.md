> Historical experiment record. Its former nonpersistent source was consolidated into `boltops/fused_moe/gluon/gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128.py` on 2026-11-12. The current entry now uses checkpoint `5a7e5f2`; old paths/configurations below describe the historical measurement. Current CK-only measurements and instruction counts are recorded in [gluon_channel_gemm1.md](gluon_channel_gemm1.md).

# 已提交基线与完整 MLS store 组合对照

基线是 `dev/gluon-scale-dot-moe-20261103` 已提交的 `92a7bfc`，在独立 worktree `BoltOPs-moe-store-compare-20261110` 复现。没有复制该分支的 staged/unstaged diff。原始 kernel 文件 SHA256 与 git show 对应提交完全一致，见实验目录 `sources.json`。

实验源码：`boltops/fused_moe/gluon/gemm_top_w8a8_channel_wasp_matrix_store_combined.py`。保留基线的 P0 完整 A/P1 B-right 分工、A3/B2、metadata 预取、C0 BF16 初始化、每次两 BK、指令预热和 MMAC 前后优先级 2/1；叠加 interleave2 MMAC/B LDS layout、转换前寄存器打包、32 列面板 scale/convert/MLS store、N 尾块过滤、四组计算交接。输出为 native GEMM1 `c_sorted=True` 的 sorted/padded 行，由后续 activation/GEMM2 消费。原 kernel 的 route-indexed 输出和默认入口未修改。

## 配置与口径

M4096、输出 N1024、K2048、E256、topk8；BM128/BN256/BK128，2P2C、每 partition 4 waves，WDRA，MLS B，SHARE_ROUTES=True，PINGPONG=True，A3/B2，WRAP32/COPY16，partition registers=(32,48,216,216)，INSN_WARMUP=True。完整单 TG，每次32768个输出；TG首 wave 到末 wave 的 PMD ticks/simFreq。两边 CPU输入及 golden 完全相同，rtol=.02、atol=.002，同时检查非目标输出仍为 NaN。LDS 都是116740B。

|版本|完整单TG μs|说明|
|---|---:|---|
|92a7bfc 提交基线|14.829815|原 route-indexed store|
|基线+完整优化|12.805520|四组交接+interleave2+sorted MLS store|

两边各三次耗时完全一致，耗时降低13.65%，速度比1.158x。重复测量及补充数值验证记录见实验目录 results.json 和 validate-combined/run.log。

## 工具链与复现

PMD b61a79c5、gfx946/64 banks/128KiB LDS；冻结 Triton `/home/zhenx/temp/moe-p0-full-a-20261110/triton` a3f89b25306c382cb8b01d1bb7d604fffdf63f04；冻结 clang `/home/zhenx/temp/moe-13us-toolchain-20261110/clang-18`，SHA256 eba870759d964c743c8fbe6dbac6213c4dbcc6c9eb11818eae26b187f171f064。所有变体使用同一工具链。

完整探针、输入 hash、编译产物、IT、stats、perf：`/home/zhenx/temp/moe-store-commit-compare-20261110/`。在容器 zx-perf-node60 中运行对应 baseline/combined 的 launch.py；宿主 /home/zhenx 映射 /zhenxin。

四组交接使用 ID8..15，每个ID由C0/C1各一个wave参与（count2），A/B READY/EMPTY计数保持整体同步。当前固定产物经运行前汇编检查：编译器生成的立即数 ebarrier ID小于8，动态交接为count2；人工检查 ID来自wave%4+8/+12。编译器仍为partition预留8/9，必须保留该产物检查，不能将固定ID分配泛化到任意调用。

此前独立实验代码及perf已备份：`/home/zhenx/temp/moe-store-refine-20261110/groups4-code-backup-20261110.tar.gz`，同目录提供SHA256清单。未改动另一任务 worktree。

## 短循环验证范围

额外短K测试最初启用原分支的 INSN_WARMUP=True，K512在汇编阶段报 `.Lmoe_iwarm_c{0,1}_b{0,1}_mmac` 重复定义：LLVM复制/展开了包含显式标签的短循环。这不是PMD数值错误。K2048三次均成功编译、执行并数值通过。为单独验证sorted输出、padding及尾块，补充短K用例显式关闭该长循环指令预热，并使用与性能相同的partition寄存器配额；原始失败日志保留在validate-bn256/run.log，未修改原分支预热代码，也未将关闭预热的结果冒充12.805520us。

短循环补充验证：关闭长循环预热的4项sorted输出/strided scales/padding/N尾块检查全部通过（26.11s），日志validate-combined-no-warmup/run.log。

## 四组交接单项消融

相同完整优化版，仅还原原来的单组8-wave交接（ID10/11、count8），其余源码变化仅交接ID/计数，interleave2、MLS B、packed N32输出、BF16初始化、预热、双BK循环、优先级、寄存器和输入全部保持不变。

|交接方式|完整TG μs（三次一致）|输出|
|---|---:|---|
|单组，每个ID8 waves|13.608595|32768项0错误，非目标输出NaN|
|四组，每个ID2 waves|12.805520|32768项0错误，非目标输出NaN|

四组相对单组减少0.803075us，耗时降低5.90%。不能把12.805520与未叠加尾块优化的14.829815的全部差额都归于四组。

循环核对使用IT中setprio2与setprio1之间的有效FP8 MMAC，排除EXEC=0预热；每个consumer wave均1024条、每SIMD2048条。SIMD0的首末有效FP8跨度含末指令8周期：单组19304周期，四组17896周期；按8周期发射窗口并集估算覆盖率84.87%→91.55%。SIMD1/2/3单组约84.86%→四组88.09%/87.95%/88.14%。这是明确窗口下的发射覆盖率估算，不是GUI利用率counter，也不能直接和不同选区的4个百分点比较。逐项数据和脚本见handoff-loop-cycles.json、analyze_handoff.py。

配对perf文件名包含HANDOFFGROUP1/HANDOFFGROUP4，均在实验目录根部；单组消融源码保留在实验目录handoff1-ablation-source.py，仅供实验，不纳入提交。

## 最终提交整合验证

整合到当前提交d2c57b1时复用原kernel公共producer/initialize/预热helpers，将整份复制缩减为259行独立入口；保留目标分支新增的短循环预热标签修复。整合后完整M4096/N1024/K2048单TG为12.906075us，32768输出零错误、非目标NaN保持、LDS116740B。该版本与早先12.805520us实验版有短循环保护差异，不能将旧perf作为最终提交的实测值。开启INSN_WARMUP的4项A2/A3、短K、stride scales、padding与N尾块回归全部通过（26.56s），含汇编ebarrier ID检查。日志integration/run.log、validate-integration/run.log。原route-indexed入口及默认MoE调度不变，新入口需显式选择sorted输出。

## 2026-11-11：小 MLS 面板，collective 交接

B 改用 K64/N32 interleave2 搬运组，由两条 matrix_load_64x16_b8 构成，依赖 Triton 提交 `b54a0b06daece337870b9f0edbb30d687761de4c`。consumer 使用 TN2、WM2/WN2；输出按32x64 CTA面板流式写出，避免重新转TN4引入LDS交换。交接使用ID10/11、count8，不包含逐SIMD交接。

保持上述 M4096/N1024/K2048、BM128/BN256/BK128、A3/B2、WRAP32 等配置，完整单TG三次均12.401025us；每次32768输出零误差、非目标NaN保持。开启预热的6项回归通过，覆盖BK64/128、BN128/256、A2/3、MLS/buffer、N尾块、非零tile/expert与无效route。原API的abarrier_wait调用同步为当前Triton的abarrier_try_wait。

每consumer wave/2BK，B matrix read32→16，A read2st64 8→16，LDS指令40→32，MMAC仍128条。此时对照183ff6d为12.906075us。原始日志和冻结源码：`/home/zhenx/temp/moe-small-interleave2-20261111/`，其中small-handoff1及small-repeat1/2为本笔对应测试，pytest-handoff1.log记录6项回归。

## 2026-11-11：四组逐 SIMD 交接，ID 外提

在小面板基础上，C0/C1同SIMD的wave成对交接，每个ID count2；两个方向使用ID8..11/12..15。wave ID和两个barrier ID在K循环前计算一次，READY/EMPTY协议不变。

每consumer wave/2BK，SALU48→20（少28条），VALU仍8，LDS32和MMAC128均不变。完整单TG三次均12.194000us，比本页上一版12.401025us减少1.67%；相同6项回归通过。循环内按8cycle MMAC窗口并集估算的覆盖率，SIMD0/1/2/3为96.99/94.81/94.95/94.90%，不是整个TG的硬件利用率counter。

测试证据：上述目录的small-hoist、small-hoist-repeat1/2、pytest-hoist.log、hoist-instruction-counts.json及hoist-loop-cycles.json。perf内嵌HSACO与保存产物一致。动态交接ID仍不能与编译器预留8/9混用；测试检查当前产物没有该冲突。

## 2026-11-11：非对称 ping/pong 交接，warmup 默认开启

保留四组逐 SIMD 配对（ping ID8..11、pong ID12..15，count2），在原两 BK 循环内改用 TileLang persistent GEMM 的定向交接：

| Consumer | ping | pong |
|---|---|---|
| C0 | 读完 LDS，arrive(ping)，priority1，MMAC，priority0 | 读完 LDS，sync(pong)，priority3，MMAC，priority0 |
| C1 | sync(ping)，读取 LDS，priority0，MMAC，再 arrive(pong) | 读完 LDS，priority2，MMAC，priority0 |

删除原先 C1 循环前的启动通知；不再每个 BK 都等待对方上一块 MMAC 完成。通知、等待和 MMAC 保留调度保护。producer、A3/B2 ring、scale、MFMA/MLS layout、输出接口不变；只修改 matrix_store_combined 入口，原 route-indexed kernel 不改。

INSN_WARMUP 仍默认 True，未采用分散 SIMD 预热实验。预热从符号标签直接进入 MMAC 块，EXEC=0；跳过新 barrier/priority setup，并在 priority reset/C1 arrive/ring 更新前返回。已有不透明循环起点保护保留。回归同时覆盖开关 warmup，检查每个标签恰好一个、预热区间没有 barrier/priority/wait/普通跳转。

M2048/N1024/K2048、BM128/BN256/BK128、W4/WDRA、MLS64x32/TN2/WM2WN2、SHARE1、WRAP32/COPY16、REG32/48/216/216，完整单 TG：原交接 warmup 开/关为12.000170/12.755470us，新交接关/开为12.491570/11.834095us。开启 warmup 的新交接比原基线快1.38%，16640项CPU golden/D2H零错误，LDS116740B。单次新配置计时；不将循环区间覆盖率当作整个TG利用率。

兼容性：12项回归全部通过（29.10s），覆盖开关warmup、BK64/128、BN128/256、A2/A3、MLS/Buffer、短循环、stride scales、padding/N尾块；另K1024/3072/4096在开启warmup时编译、执行及数值对照通过，四个预热标签无重复。受既有约束限制：K须为2*BK的倍数，当前gfx946配置；不是任意LLVM版本或任意形状的保证。

实验源码/完整产物、数值、IT/stats/perf、回归日志：`/home/zhenx/temp/moe-tilelang-handoff-20261111/`。warm-result.json是新交接+warmup的逐SIMD审计，compatibility.log、warm-large-k-check.json是兼容性记录。
