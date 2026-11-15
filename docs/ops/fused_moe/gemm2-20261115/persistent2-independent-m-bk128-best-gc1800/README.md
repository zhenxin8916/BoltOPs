# BK128 最好：15.282772 μs

与本次源码和证据一起提交，具体commit见本目录git log。1.8GHz、48CU、每persistent TG两个独立M输出，共96输出；N2048/K256/BM128/BN256/BK128。2190336元素零超差（rtol .02/atol .002），全48CU dispatch MMAC14.901590%（精确值见model-metrics.json）。同HSACO、源码SHA、输入SHA和GPU地址复测完全一致。比父版264d7e7的15.327252μs降低0.2902%；20%目标尚未达成。

2P2C、117KiB LDS、39/9/174/174实际VGPR、无scratch spill。A完整32KiB，Bleft单16KiB、Bright双16KiB，两个BF16输出槽各16KiB，MA双M槽4KiB，SB1KiB。P0发A并搬SA/weight/route/expert到LDS、散写BF16；P1发BL0/BR0/BR1/BL1并搬SB到LDS。消费者各128列，K0按两个64行面板计算，K1按四个32行面板计算和缩放，两块合并发布一次64行输出。FP32仍严格((acc*SA)*SB)*weight，再转BF16，buffer_store_dwordx4散写。

唯一算法变动：原先A LDS读→wait0→A EMPTY3→首块B/SB LDS读→wait0→B EMPTY，改为A LDS读→等待B READY→B/SB LDS读→一次wait0→A EMPTY3→B EMPTY。所有读取都完成后才归还两个物理槽，没有提前释放A。MA元数据仍在K0计算后读取。没有增加硬件同步，没有手写ebarrier。P0仅A发射窗口priority2，A READY后恢复0，P1保持0；无额外C0/C1计算token。

audit_combined_ds_wait.py按真实IT核对8消费者wave各两个M：A16条加首B8条matrix DS读、SB一条ds_read2_b64都在唯一wait0之前，A EMPTY在wait0之后。A/B真实代际、16waves、2048有效MMAC/CU、exec0=0、忙8cycle并集、16ID范围/首wait非0/无相邻同ID、消费者0VMEM/0permlane及A priority窗口均通过。

CU0 SIMD0跨M计算gap1596/2060→1460/1280cycle，最后MMAC至最终READY708/604cycle。首次C0 MMAC晚24cycle，C1早420cycle；局部并非全部改善，结论依据全dispatch总耗时。

固定N遍历截为1的两独立M冻结诊断；正式默认kernel未替换，完整8N/3008输出未由本次测试覆盖。工具链二进制SHA逐项复核与264d7e7一致；LLVM分支commit/本地tracked补丁、Triton分支commit/本地修改、冻结runtime和PMD信息见toolchain-provenance.json及同目录补丁。无新工具链修改。
