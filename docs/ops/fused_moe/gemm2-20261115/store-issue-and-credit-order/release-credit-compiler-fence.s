	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx946"
	.globl	release_credit                  ; -- Begin function release_credit
	.p2align	8
	.type	release_credit,@function
release_credit:                         ; @release_credit
; %bb.0:                                ; %entry
	s_load_dword s0, s[4:5], 0x0
	s_load_dword s1, s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, s0
	ds_read_b32 v0, v0
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_bitcmp0_b32 s1, 0
	s_waitcnt lgkmcnt(0)
	s_abarrier_arrive  s0, 11, 1
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %store
	s_load_dwordx2 s[0:1], s[4:5], 0x8
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	global_store_dword v1, v0, s[0:1]
.LBB0_2:                                ; %exit
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel release_credit
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_matrix_excp 0
		.amdhsa_cancellation_excp 0
		.amdhsa_vgpr_alloc_mode 0
		.amdhsa_bf16_overflow 0
		.amdhsa_bf8_overflow 0
		.amdhsa_fp8_overflow 0
		.amdhsa_bf16_denorm_mode 3
		.amdhsa_scale_buffer_size 0
		.amdhsa_tf32_overflow 0
		.amdhsa_tf32_denorm_mode 3
		.amdhsa_bf16_rounding_mode 0
		.amdhsa_uc_cc_ack_early_return 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	release_credit, .Lfunc_end0-release_credit
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 76
; NumSgprs: 10
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; LatencyBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 2
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX938:BF16_OVFL: 0
; COMPUTE_PGM_RSRC3_GFX938:BF8_OVFL: 0
; COMPUTE_PGM_RSRC3_GFX938:FP8_OVFL: 0
; COMPUTE_PGM_RSRC3_GFX938:BF16_DENORM: 3
; COMPUTE_PGM_RSRC3_GFX946:SCALE_BUFFER_SIZE: 0
; COMPUTE_PGM_RSRC3_GFX946:TF32_OVFL: 0
; COMPUTE_PGM_RSRC3_GFX946:TF32_DENORM: 3
; COMPUTE_PGM_RSRC3_GFX946:BF16_RMODE: 0
; COMPUTE_PGM_RSRC3_GFX946:UC_CC_ACK_EARLY_RETURN: 1
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  local
        .name:           shared
        .offset:         0
        .pointee_align:  1
        .size:           4
        .value_kind:     dynamic_shared_pointer
      - .address_space:  global
        .name:           out
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .name:           write
        .offset:         16
        .size:           1
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .max_flat_workgroup_size: 1024
    .name:           release_credit
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         release_credit.kd
    .unfolded_args:
      - .address_space:  local
        .name:           shared
        .offset:         0
        .pointee_align:  1
        .size:           4
        .value_kind:     dynamic_shared_pointer
      - .address_space:  global
        .name:           out
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .name:           write
        .offset:         16
        .size:           1
        .value_kind:     by_value
    .uses_dynamic_stack: false
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx946
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
