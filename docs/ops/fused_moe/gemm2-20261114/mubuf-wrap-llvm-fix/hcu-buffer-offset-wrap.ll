; RUN: llc -mtriple=amdgcn -mcpu=gfx946 -verify-machineinstrs < %s | FileCheck %s

; Buffer offsets are i32 arithmetic. Folding an immediate must not turn
; a wrapping IR add into a widened hardware range-check addition.

declare <2 x float> @llvm.amdgcn.raw.ptr.buffer.load.v2f32(ptr addrspace(8), i32, i32, i32 immarg)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float>, ptr addrspace(8), i32, i32, i32 immarg)

; CHECK-LABEL: wrapping_offset:
; CHECK: v_add_u32{{.*}}0xf00
; CHECK: buffer_load_dwordx2 {{.*}} offen{{$}}

define amdgpu_ps <2 x float> @wrapping_offset(ptr addrspace(8) inreg %rsrc, i32 %base) {
  %offset = add i32 %base, 3840
  %v = call <2 x float> @llvm.amdgcn.raw.ptr.buffer.load.v2f32(ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret <2 x float> %v
}
; CHECK-LABEL: wrapping_aligned_offset:
; CHECK: v_add_u32{{.*}}0x1e00
; CHECK: buffer_load_dwordx2 {{.*}} offset:256

define amdgpu_ps <2 x float> @wrapping_aligned_offset(ptr addrspace(8) inreg %rsrc, i32 %raw) {
  %base = and i32 %raw, -512
  %offset = add i32 %base, 7936
  %v = call <2 x float> @llvm.amdgcn.raw.ptr.buffer.load.v2f32(ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret <2 x float> %v
}
; CHECK-LABEL: aligned_no_wrap:
; CHECK: buffer_load_dwordx2 {{.*}} offset:3840

define amdgpu_ps <2 x float> @aligned_no_wrap(ptr addrspace(8) inreg %rsrc, i32 %raw) {
  %base = and i32 %raw, -4096
  %offset = add i32 %base, 3840
  %v = call <2 x float> @llvm.amdgcn.raw.ptr.buffer.load.v2f32(ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret <2 x float> %v
}
; CHECK-LABEL: nuw_offset:
; CHECK: buffer_load_dwordx2 {{.*}} offset:3840

define amdgpu_ps <2 x float> @nuw_offset(ptr addrspace(8) inreg %rsrc, i32 %base) {
  %offset = add nuw i32 %base, 3840
  %v = call <2 x float> @llvm.amdgcn.raw.ptr.buffer.load.v2f32(ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret <2 x float> %v
}
; CHECK-LABEL: wrapping_store:
; CHECK: v_add_u32{{.*}}0xf00
; CHECK: buffer_store_dwordx2 {{.*}} offen{{$}}

define amdgpu_ps void @wrapping_store(ptr addrspace(8) inreg %rsrc, i32 %base, <2 x float> %v) {
  %offset = add i32 %base, 3840
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2f32(<2 x float> %v, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}
