	.text
	.section	.AMDGPU.config,"",@progbits
	.long	45096
	.long	0
	.long	165608
	.long	0
	.long	45100
	.long	0
	.long	165580
	.long	1
	.long	165584
	.long	1
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.globl	wrapping_offset                 ; -- Begin function wrapping_offset
	.p2align	8
	.type	wrapping_offset,@function
wrapping_offset:                        ; @wrapping_offset
; %bb.0:
	v_add_u32_e32 v0, 0xf00, v0
	buffer_load_dwordx2 v[0:1], v0, s[0:3], 0 offen
	s_waitcnt vmcnt(0)
	; return to shader part epilog
.Lfunc_end0:
	.size	wrapping_offset, .Lfunc_end0-wrapping_offset
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 20
; NumSgprs: 8
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; LatencyBound: 0
; FloatMode: 240
; IeeeMode: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 2
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 0
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 0
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
	.section	.AMDGPU.config,"",@progbits
	.long	45096
	.long	0
	.long	165608
	.long	0
	.long	45100
	.long	0
	.long	165580
	.long	1
	.long	165584
	.long	1
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.globl	wrapping_aligned_offset         ; -- Begin function wrapping_aligned_offset
	.p2align	8
	.type	wrapping_aligned_offset,@function
wrapping_aligned_offset:                ; @wrapping_aligned_offset
; %bb.0:
	v_and_b32_e32 v0, 0xfffffe00, v0
	v_add_u32_e32 v0, 0x1e00, v0
	buffer_load_dwordx2 v[0:1], v0, s[0:3], 0 offen offset:256
	s_waitcnt vmcnt(0)
	; return to shader part epilog
.Lfunc_end1:
	.size	wrapping_aligned_offset, .Lfunc_end1-wrapping_aligned_offset
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 28
; NumSgprs: 8
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; LatencyBound: 0
; FloatMode: 240
; IeeeMode: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 2
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 0
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 0
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
	.section	.AMDGPU.config,"",@progbits
	.long	45096
	.long	0
	.long	165608
	.long	0
	.long	45100
	.long	0
	.long	165580
	.long	1
	.long	165584
	.long	1
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.globl	aligned_no_wrap                 ; -- Begin function aligned_no_wrap
	.p2align	8
	.type	aligned_no_wrap,@function
aligned_no_wrap:                        ; @aligned_no_wrap
; %bb.0:
	v_and_b32_e32 v0, 0xfffff000, v0
	buffer_load_dwordx2 v[0:1], v0, s[0:3], 0 offen offset:3840
	s_waitcnt vmcnt(0)
	; return to shader part epilog
.Lfunc_end2:
	.size	aligned_no_wrap, .Lfunc_end2-aligned_no_wrap
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 20
; NumSgprs: 8
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; LatencyBound: 0
; FloatMode: 240
; IeeeMode: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 2
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 0
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 0
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
	.section	.AMDGPU.config,"",@progbits
	.long	45096
	.long	0
	.long	165608
	.long	0
	.long	45100
	.long	0
	.long	165580
	.long	1
	.long	165584
	.long	1
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.globl	nuw_offset                      ; -- Begin function nuw_offset
	.p2align	8
	.type	nuw_offset,@function
nuw_offset:                             ; @nuw_offset
; %bb.0:
	buffer_load_dwordx2 v[0:1], v0, s[0:3], 0 offen offset:3840
	s_waitcnt vmcnt(0)
	; return to shader part epilog
.Lfunc_end3:
	.size	nuw_offset, .Lfunc_end3-nuw_offset
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12
; NumSgprs: 8
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; LatencyBound: 0
; FloatMode: 240
; IeeeMode: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 2
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 0
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 0
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
	.section	.AMDGPU.config,"",@progbits
	.long	45096
	.long	0
	.long	165608
	.long	0
	.long	45100
	.long	0
	.long	165580
	.long	3
	.long	165584
	.long	3
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.globl	wrapping_store                  ; -- Begin function wrapping_store
	.p2align	8
	.type	wrapping_store,@function
wrapping_store:                         ; @wrapping_store
; %bb.0:
	v_add_u32_e32 v0, 0xf00, v0
	buffer_store_dwordx2 v[1:2], v0, s[0:3], 0 offen
	s_endpgm
.Lfunc_end4:
	.size	wrapping_store, .Lfunc_end4-wrapping_store
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 20
; NumSgprs: 8
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; LatencyBound: 0
; FloatMode: 240
; IeeeMode: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 3
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 0
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 0
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
	.amd_amdgpu_isa "amdgcn----gfx946"
