# BN512/BK64 当前最好：21.981460 μs

1.8GHz、48CU、48个物理persistent TG，每TG两个独立M输出tile（每个128×512），实际N2048/K256，诊断只算前512列。4380672有效元素零超差（rtol.02/atol.002）；全48CU整个dispatch MMAC20.720880%。同HSACO、源码、输入SHA、GPU地址复测完全一致，见repeat-proof.json及repeat目录。仓库文件只修正了一行AB映射注释，其Python AST与实测源码完全一致；两个SHA均记录，归档源码保留实测原文。

比首个有效BN512基线22.131580μs减少0.6783%。唯一算法改动是P1发射顺序从BL0 BR0 BR1 BL1 BR2 BL2 BR3 BL3改为BL0 BR0 BR1 BL1 BL2 BR2 BL3 BR3；槽位、同步协议、优先级、缩放和写出布局均不变。CU0 SIMD0 C1跨M空档1916→1624cycle，C0为1116→1444cycle，整体收益不能理解为每处等待都缩短。

2P2C各4wave。P0加载完整A32KiB及SA/weight/route/expert元数据，预发下一M A，读取消费者BF16结果并以buffer_store_dwordx4散写。P1加载B和SB。C0/C1分别负责256列，每组拆成两个N128阶段，A在寄存器跨阶段复用；K0..K2按M64计算，K3按M32计算、缩放和发布。FP32顺序((acc*SA)*SB)*weight保持。85KiB LDS：A32、BL双8、BR双8、输出双8、MA4、SB1。实际VGPR40/9/187/187，无spill。

BL READY5/10 EMPTY6/14；BR READY7/2 EMPTY8/4；K0..3 phase0,0,1,1。AB10是BL第二槽READY。A READY1/EMPTY3须两个消费者各4wave共同归还；A仅在当前M第一个N阶段捕获一次并归还一次。真实IT审计16wave、4096有效MMAC/CU、exec0=0、8cycle并集、16ID范围/首wait非0/无相邻同ID、A/B代际、消费者0VMEM/0permlane、A-only priority2窗口全部通过。

与冻结8836708的BN256等填充计算量对照：BN256每TG四个输出，共192tile，对比BN512每TG两个输出，共96tile；均3221225472 padded FLOPs、196608条MMAC、同一完整输入SHA。BN256为23.493780μs/19.387055%，BN512为21.981460μs/20.720880%，耗时减少6.437108%。前者检查4330752有效元素、后者4380672，行/列前缀不同，不是相同输出集合或完整N2048验收；20%目标尚未达到。BN256源码及9656字节.text机器码保持原样，HSACO调试路径不同。详见../bn512-bk64-study/equal-work-proof.json。

工具链沿用8836708冻结MLS runtime，未使用宽A诊断修补。LLVM/Triton分支、commit、本地修改及二进制SHA见inherited-toolchain-provenance.json，补丁实体沿用已提交的../persistent2-independent-m-bk128-best-gc1800/归档。PMD b61a79c5；libtriton SHA256 ddfa0f4ab5d93ffdb99f4fbd367d6a1f42a2ffec0f8f44545846b2983dc95dd4。perf路径及SHA见perf.json；源码、HSACO、编译IR/metadata、命令、输入记录和审计脚本随报告保存。容器/zhenxin映射宿主/home/zhenx。
