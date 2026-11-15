# BN512 大 A MLS 对照（隔离诊断）

1.8GHz，48CU，2P2C，每persistent TG两个独立M输出，BM128/BN512/BK64/K256，共96个128×512输出，实际N2048中前512列。每例4380672有效元素零超差。正式kernel和冻结工具链未替换。

| A MLS及发射布局 | 耗时μs | 全48CU整个dispatch MMAC | 两M的P0 MLS/CU | 两M的消费者A DS/CU |
|---|---:|---:|---:|---:|
| 64×16 b8，4×1，原最好 |21.981460|20.720880%|64|256|
| 128×16 b8，4×1 |22.589724|20.162938%|64|256|
| 128×16 b8，2×2 |21.883048|20.814066%|32|256|

2×2版本同HSACO/源码/完整输入SHA/GPU地址复测耗时一致，比原最好减少0.447705%。计算量和输出集合完全一致，MMAC 196608条，CU0 16waves，exec0=0，8cycle MMAC忙区并集、A/B phase、AB ID硬件约束、消费者0VMEM以及P0 A-only priority2均通过。LDS仍85KiB，实际VGPR40/9/187/187，无spill。

红框是A从LDS到REG的读取：每个消费者wave每M读取64个VGPR，原版及宽A版均16条ds_read_matrix_trans_format，每条4个VGPR/128bit。Builtin表中u8返回intx4，LLVM DS_MAT_OPERAND为HCU_VReg_128；当前接口没有更宽的单条返回。更大的matrix_load_128x16_b8改变global→LDS，不改变这一REG读宽。原版CU0 SIMD0第一M的C0/C1 A DS发射跨度64/276cycle；大A 2×2是88/324，局部红框没有缩短，不能将总耗时收益描述成减少DS或每处等待均改善。

更宽A使用已有隔离unit-M16行选择修正，补丁及源码见comparison.json引用，原库未更改。隔离库在原kernel上的.text SHA与原冻结工具链相同。该修正的一般布局/类型回归尚未建立，故保留为试验结果，不替换正式kernel，也不把本试验称为可直接使用原冻结工具链编译的版本。运行源、HSACO、IR、命令、结果、审计及实际perf路径均归档；不保存变慢4×1版本的实现。
