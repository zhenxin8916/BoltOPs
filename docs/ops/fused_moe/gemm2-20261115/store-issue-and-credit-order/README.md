# Store 发射与 EMPTY 归还顺序

最优基线为0054799，1.8GHz/48CU/每TG两个独立M输出，16.374200μs。store-issue-analysis.json按IT记录相邻发射时间，包含调度及依赖，不当作单条指令硬件延迟；GM credit计数包括load、MLS和store，不能全部归给store。

P0 SIMD0的READY等待：M0依次2264/1272/4/88cycle，M1依次1448/380/4/4cycle。每组四条x4 store的首末跨度84~196cycle，通常每wave约64cycle一条；合并四SIMD后通常13~17cycle一条。CU0 GM cmd/addr/data credit stall为422/32/0，TA send stalled by TCP为576cycle；这些支持存在写出输送压力，但更大的空档来自消费者结果未到。

尝试条件跳过完全填充的64行输出时，2190336元素中187678失败。同HSACO复测一致，CPU已验证专家内有效行前缀、填充行在尾部。查看LLIR，LDS读取在EMPTY arrive之前；查看生成汇编及IT，ds_read被后端下沉到条件store分支，位于EMPTY归还之后，形成下一代覆写竞争。不是PMD模型归因。

纯LLVM最小例release-credit.ll可由llc -march=amdgcn -mcpu=gfx946 -O3编译：原版汇编arrive在ds_read之前；加入空asm memory clobber的对照保持ds_read在arrive之前，无新增硬件同步。完整流水空asm按BF16 pack2绑寄存器后数值0错误，但耗时16.681112μs，未超过最好，不保留其优化代码。

进一步定位为MachineInstr::isSafeToMove：hasSideEffects阻止barrier自身移动，但不设置MachineSink的SawStore，因此此前LDS读取仍可越过释放点下沉。第一版将abarrier标为mayLoad/mayStore虽然修正读取顺序，却令已有abarrier_wait_branch.ll新增buffer_wbinvl1_vol等操作，已撤回，不能视为最终修复。

收窄版本在SIInstrInfo::shouldSink中禁止将后面存在LHBarrier的load下沉出当前基本块；不将barrier伪装成真实访存，SALU/VALU地址计算仍可下沉。新回归abarrier-memory-order.ll覆盖条件写出前释放EMPTY的最小例，已重建llc/clang；新增最小回归、原abarrier_wait_branch.ll、hcu-control-only-ebarrier-sync.mir均通过（3/3），完整流水两版均2190336元素零超差，16waves/2048有效MMAC/exec0=0/16ID/首wait非0/无相邻重复审计通过。既有hcu-matrix-store-lds-waitcnt.mir在原始llc也失败，单独记录为既有失败，不修改其检查。历史最好二进制不替换。

原始clang已备份为/zhenxin/temp/llvm-abarrier-memory-order-20261115/original-clang-18；对应original-clang-control-sync包装器含-B原build/bin以定位ld.lld。旧最好需使用该原始编译器复现，不能把新的全局build路径当作原二进制版本。

最终复测：修复后原最好control为16.374200μs/13.908319185%，与原编译器最好耗时完全一致；条件跳过无效64行panel为16.537108μs/13.771307535%，慢0.995%，不晋升。两版ISA均无buffer_wbinvl1。修复不要求kernel空asm或额外硬件同步，旧最好算法不变。LLVM修复留在工作区diff，未混入既有暂存的其他LLVM变更；GEMM2未生成新的性能胜出版commit。

非Gluon参考为boltops/fused_moe/triton/moe_op.py的block_n_diviable/block_k_diviable与末尾c_mask选择（当前约2353行）。Gluon当前整除N/K已裁剪相应mask；专家路由padding是运行时数据，不能直接按M/N/K常量删除。EXPERTS_LOCAL常量对照仅在identity专家映射且无非本地expert时合法，不可默认用于EP输入。本对照16.870152μs，亦不晋升。

独立abarrier-memory-order.patch采用零上下文格式，可在对应LLVM HEAD上用git apply --unidiff-zero应用；最好归档另包含完整tracked LLVM本地补丁。
