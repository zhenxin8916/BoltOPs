; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx946 -amdgpu-codegenprepare -verify-each %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx946 -O3 < %s | FileCheck %s --check-prefix=GCN
;
; A wrapped LDS index can expose a constant byte offset only if known bits
; prove no wrap/carry through the masked field. Keep real wraps non-affine.
target datalayout = "e-p:64:64-p1:64:64-p3:32:32-i64:64-n32:64"
@lds = internal addrspace(3) global [65536 x i8] undef, align 65536
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @no_wrap_field(ptr addrspace(1) %out) {
; IR-LABEL: @no_wrap_field(
; IR: %lds.nowrap.offset
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 32
; GCN-LABEL: no_wrap_field:
; GCN-NOT: ds_read_b64
; GCN: ds_read2_b64
; GCN-NOT: ds_read_b64
; GCN: s_endpgm
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lo = and i32 %tid, 888
  %hi = and i32 %tid, 12288
  %s0 = add i32 %lo, 32
  %m0 = and i32 %s0, 1016
  %a0 = or i32 %hi, %m0
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %a0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %s1 = add i32 %lo, 64
  %m1 = and i32 %s1, 1016
  %a1 = or i32 %hi, %m1
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %a1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @no_wrap_lowmask(ptr addrspace(1) %out) {
; IR-LABEL: @no_wrap_lowmask(
; IR: %lds.nowrap.offset
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 64
; GCN-LABEL: no_wrap_lowmask:
; GCN-NOT: ds_read_b64
; GCN: ds_read2_b64
; GCN-NOT: ds_read_b64
; GCN: s_endpgm
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lo = and i32 %tid, 888
  %hi = and i32 %tid, 12288
  %s0 = add i32 %lo, 32
  %m0 = and i32 %s0, 1023
  %a0 = or i32 %hi, %m0
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %a0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %s1 = add i32 %lo, 96
  %m1 = and i32 %s1, 1023
  %a1 = or i32 %hi, %m1
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %a1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @may_wrap_field(ptr addrspace(1) %out) {
; IR-LABEL: @may_wrap_field(
; IR-NOT: %lds.nowrap.offset
; IR-NOT: %lds.offset =
; GCN-LABEL: may_wrap_field:
; GCN: ds_read_b64
; GCN: ds_read_b64
; GCN: s_endpgm
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lo = and i32 %tid, 1016
  %hi = and i32 %tid, 12288
  %s0 = add i32 %lo, 32
  %m0 = and i32 %s0, 1016
  %a0 = or i32 %hi, %m0
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %a0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %s1 = add i32 %lo, 64
  %m1 = and i32 %s1, 1016
  %a1 = or i32 %hi, %m1
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %a1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @mask_has_hole(ptr addrspace(1) %out) {
; IR-LABEL: @mask_has_hole(
; IR-NOT: %lds.nowrap.offset
; IR-NOT: %lds.offset =
; GCN-LABEL: mask_has_hole:
; GCN: ds_read_b64
; GCN: ds_read_b64
; GCN: s_endpgm
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lo = and i32 %tid, 1016
  %hi = and i32 %tid, 12288
  %s0 = add i32 %lo, 32
  %m0 = and i32 %s0, 1000
  %a0 = or i32 %hi, %m0
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %a0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %s1 = add i32 %lo, 64
  %m1 = and i32 %s1, 1000
  %a1 = or i32 %hi, %m1
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %a1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}
