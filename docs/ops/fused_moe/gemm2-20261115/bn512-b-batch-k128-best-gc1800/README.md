# BN512 B按K128批量发布：20.680420 μs

1.8GHz，48CU，48个物理persistent TG，每TG两个独立M输出，BM128/BN512/BK64/K256，共96个128×512输出。实际N2048，诊断只算前512列。4380672有效元素0超差，rtol.02/atol.002。全48CU整个dispatch MMAC22.024466%，196608条MMAC。相同源码/HSACO/输入SHA/GPU地址复测耗时一致，见repeat-proof.json。正式源码仅修正注释，与实测源码AST完全一致。

P0职责不变：A32KiB、SA/weight/route/expert、下一M的A预发、BF16 buffer_store_dwordx4散写。P1把相邻两个K64的B合成一个K128 LDS槽和一次READY发布；两消费者仍各负责256列、两个N128阶段，按四个K64计算，K3按M32缩放/写出。FP32缩放顺序((acc*SA)*SB)*weight不变。

LDS117KiB：A32、BL两个16、BR两个16、输出两个8、MA4、SB1。BL READY5/10 EMPTY6/14，BR READY7/2 EMPTY8/4；每槽每N阶段只用一次，phase=b_epoch&1。发送顺序BL(K0,K1)→BR(K0,K1)→BR(K2,K3)→BL(K2,K3)。每个消费者每N的B READY等待4→2；读取两K的数据及SB后wait_lgkmcnt0再归还EMPTY。首次捕获也覆盖完整A后才归还A EMPTY3，保持下一M安全预发。实际VGPR40/9/203/203，无spill。使用原冻结MLS runtime ddfa0f4a，无宽A诊断修正。

原K64发布21.981460μs→20.680420μs（减少5.918806%）；比隔离大A21.883048μs减少5.495706%。相同填充工作量的BN256/8836708四输出为23.493780μs，本版降低11.975%，20%目标尚未达到。BN256与BN512诊断行/列前缀不同，不能外推成完整N2048验收。

16waves、exec0=0、MMAC8cycle并集、AB ID范围/首wait非0/无相邻同ID、实际A/B phase、消费者0VMEM、A-only priority2窗口均通过。CU0 SIMD0跨M空档C0由1444降1212cycle，C1由1624增2532cycle；总耗时收益来自K内发布/等待减少，不能称每处等待都改善。下一步仍需关注跨M衔接。

对照中BL第二批先发为21.189716μs，BR第二批先发为20.680420μs。A拆K128读取、先计算K0再读后半为22.650328μs，立即读后半对照22.667564μs；未采用，不保存失败实现。工具链分支/commit、本地补丁及二进制来源沿用repeat-proof.json中的已提交归档；精确源码、HSACO、IR、执行命令、输入记录和审计脚本随本报告保存。
