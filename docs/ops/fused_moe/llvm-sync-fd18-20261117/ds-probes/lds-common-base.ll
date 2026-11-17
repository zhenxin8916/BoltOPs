; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx946 -amdgpu-codegenprepare -verify-each %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx946 -O3 < %s | FileCheck %s --check-prefix=GCN
;
; Recover constant differences through wrapped LDS address def-use chains.
; The pointer rewrite must not move memory accesses across stores/barriers.

target datalayout = "e-p:64:64-p1:64:64-p3:32:32-i64:64-n32:64"
@lds = internal addrspace(3) global [65536 x i8] undef, align 65536

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()

define amdgpu_kernel void @wrapped_overlap(ptr addrspace(1) %out) {
; IR-LABEL: @wrapped_overlap(
; IR: %v0 = load i64, ptr addrspace(3) %p0
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 1024
; IR-NEXT: %v1 = load i64, ptr addrspace(3) %lds.offset
; GCN-LABEL: wrapped_overlap:
; GCN-NOT: ds_read_b64
; GCN: ds_read2{{(st64)?}}_b64
; GCN-NOT: ds_read_b64
; GCN: s_endpgm
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %x = and i32 %tid, 13208
  %y = and i32 %tid, 24
  %y32 = or i32 %y, 32
  %sum0 = add i32 %x, %y32
  %lo0 = and i32 %sum0, 1016
  %hi0 = and i32 %x, 12288
  %a0 = getelementptr i8, ptr addrspace(3) @lds, i32 %hi0
  %p0 = getelementptr i8, ptr addrspace(3) %a0, i32 %lo0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %xc = or i32 %x, 1056
  %sum1 = add i32 %xc, %y
  %lo1 = and i32 %sum1, 1023
  %hi1 = and i32 %xc, 13312
  %a1 = getelementptr i8, ptr addrspace(3) @lds, i32 %hi1
  %p1 = getelementptr i8, ptr addrspace(3) %a1, i32 %lo1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @wrapped_disjoint(ptr addrspace(1) %out) {
; IR-LABEL: @wrapped_disjoint(
; IR: %v0 = load i64, ptr addrspace(3) %p0
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 2048
; IR-NEXT: %v1 = load i64, ptr addrspace(3) %lds.offset
; GCN-LABEL: wrapped_disjoint:
; GCN-NOT: ds_read_b64
; GCN: ds_read2{{(st64)?}}_b64
; GCN-NOT: ds_read_b64
; GCN: s_endpgm
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %x = and i32 %tid, 14232
  %y = and i32 %tid, 64
  %xy = or disjoint i32 %x, %y
  %sum0 = add i32 %xy, 32
  %lo0 = and i32 %sum0, 1016
  %hi0 = and i32 %x, 13312
  %a0 = getelementptr i8, ptr addrspace(3) @lds, i32 %hi0
  %p0 = getelementptr i8, ptr addrspace(3) %a0, i32 %lo0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %xc = or i32 %x, 2080
  %sum1 = add i32 %xc, %y
  %lo1 = and i32 %sum1, 1023
  %hi1 = and i32 %xc, 15360
  %a1 = getelementptr i8, ptr addrspace(3) @lds, i32 %hi1
  %p1 = getelementptr i8, ptr addrspace(3) %a1, i32 %lo1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @typed_gep(ptr addrspace(1) %out, i32 %idx, ptr addrspace(3) %alias) {
; IR-LABEL: @typed_gep(
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 1024
; IR: ret void
  %a = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @lds, i32 2048), i32 %idx
  %p0 = getelementptr [0 x i8], ptr addrspace(3) %a, i32 0, i32 4096
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %b = getelementptr i8, ptr addrspace(3) @lds, i32 %idx
  %p1 = getelementptr [0 x i8], ptr addrspace(3) %b, i32 0, i32 7168
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @negative_delta(ptr addrspace(1) %out, i32 %idx, ptr addrspace(3) %alias) {
; IR-LABEL: @negative_delta(
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 -1024
; IR: ret void
  %a = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @lds, i32 2048), i32 %idx
  %p0 = getelementptr [0 x i8], ptr addrspace(3) %a, i32 0, i32 4096
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %b = getelementptr i8, ptr addrspace(3) @lds, i32 %idx
  %p1 = getelementptr [0 x i8], ptr addrspace(3) %b, i32 0, i32 5120
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @volatile_load(ptr addrspace(1) %out, i32 %idx, ptr addrspace(3) %alias) {
; IR-LABEL: @volatile_load(
; IR-NOT: %lds.offset
; IR: ret void
  %a = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @lds, i32 2048), i32 %idx
  %p0 = getelementptr [0 x i8], ptr addrspace(3) %a, i32 0, i32 4096
  %v0 = load volatile i64, ptr addrspace(3) %p0, align 8
  %b = getelementptr i8, ptr addrspace(3) @lds, i32 %idx
  %p1 = getelementptr [0 x i8], ptr addrspace(3) %b, i32 0, i32 7168
  %v1 = load volatile i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @unknown_or(ptr addrspace(1) %out, i32 %idx, i32 %off) {
; IR-LABEL: @unknown_or(
; IR-NOT: %lds.offset
; IR: ret void
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %idx
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %other = or i32 %idx, 1056
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %other
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @variable_offset(ptr addrspace(1) %out, i32 %idx, i32 %off) {
; IR-LABEL: @variable_offset(
; IR-NOT: %lds.offset
; IR: ret void
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %idx
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %other = add i32 %idx, %off
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %other
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @barrier(ptr addrspace(1) %out, i32 %idx, ptr addrspace(3) %alias) {
; IR-LABEL: @barrier(
; IR: %v0 = load
; IR: call void @llvm.amdgcn.s.barrier()
; IR: %v1 = load
; GCN-LABEL: barrier:
; GCN-NOT: ds_read2
; GCN: ds_read_b64
; GCN: s_barrier
; GCN: ds_read_b64
; GCN-NOT: ds_read2
; GCN: s_endpgm
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %idx
  %next = add i32 %idx, 1024
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %next
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  call void @llvm.amdgcn.s.barrier()
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @aliasing_store(ptr addrspace(1) %out, i32 %idx, ptr addrspace(3) %alias) {
; IR-LABEL: @aliasing_store(
; IR: %v0 = load
; IR: store i64 0, ptr addrspace(3) %alias, align 8
; IR: %v1 = load
; GCN-LABEL: aliasing_store:
; GCN-NOT: ds_read2
; GCN: ds_read_b64
; GCN: ds_write_b64
; GCN: ds_read_b64
; GCN-NOT: ds_read2
; GCN: s_endpgm
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %idx
  %next = add i32 %idx, 1024
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %next
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  store i64 0, ptr addrspace(3) %alias, align 8
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @mask_with_hole(ptr addrspace(1) %out, i32 %idx, i32 %y) {
; IR-LABEL: @mask_with_hole(
; IR: %lo1 = and i32 %sum1, 1535
; IR-NOT: %lds.offset
; IR: ret void
  %x = and i32 %idx, 13208
  %sum0 = add i32 %x, %y
  %lo0 = and i32 %sum0, 1535
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %lo0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %xc = or i32 %x, 1056
  %sum1 = add i32 %xc, %y
  %lo1 = and i32 %sum1, 1535
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %lo1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @overlapping_or(ptr addrspace(1) %out, i32 %idx, i32 %y) {
; IR-LABEL: @overlapping_or(
; IR: %lo1 = and i32 %sum1, 1023
; IR-NOT: %lds.offset
; IR: ret void
  %x = add i32 %idx, 0
  %sum0 = add i32 %x, %y
  %lo0 = and i32 %sum0, 1023
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %lo0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %xc = or i32 %x, 1056
  %sum1 = add i32 %xc, %y
  %lo1 = and i32 %sum1, 1023
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %lo1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

; The high term is removable only when above EVERY retained bit.
define amdgpu_kernel void @masked_reassociated_sum(ptr addrspace(1) %out, i32 %idx, i32 %wave) {
; IR-LABEL: @masked_reassociated_sum(
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 1024
; IR: ret void
  %lo = and i32 %idx, 408
  %hi = and i32 %idx, 12288
  %y = and i32 %wave, 48
  %sum = add i32 %lo, %y
  %sum0 = add i32 %sum, 32
  %offset0 = and i32 %sum0, 1016
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %offset0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %x = or disjoint i32 %lo, %hi
  %y32 = add i32 %y, 32
  %sum1 = add i32 %x, %y32
  %offset1 = and i32 %sum1, 1016
  %base1 = getelementptr i8, ptr addrspace(3) @lds, i32 1024
  %p1 = getelementptr i8, ptr addrspace(3) %base1, i32 %offset1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

; The high term is removable only when above EVERY retained bit.
define amdgpu_kernel void @masked_carry_from_hole(ptr addrspace(1) %out, i32 %idx, i32 %wave) {
; IR-LABEL: @masked_carry_from_hole(
; IR-NOT: %lds.offset
; IR: ret void
  %lo = and i32 %idx, 408
  %hi = and i32 %idx, 512
  %y = and i32 %wave, 48
  %sum = add i32 %lo, %y
  %sum0 = add i32 %sum, 32
  %offset0 = and i32 %sum0, 1535
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %offset0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %x = or disjoint i32 %lo, %hi
  %y32 = add i32 %y, 32
  %sum1 = add i32 %x, %y32
  %offset1 = and i32 %sum1, 1535
  %base1 = getelementptr i8, ptr addrspace(3) @lds, i32 1024
  %p1 = getelementptr i8, ptr addrspace(3) %base1, i32 %offset1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

; A loop-invariant variable OR hides the row constant in one operand. The
; low/high fields are disjoint, so both addresses have the same masked sum.
; Keep the dynamic slot base in the loop and share only its per-lane offset.
define amdgpu_kernel void @loop_nested_disjoint_or(ptr addrspace(1) %out, i32 %count) {
; IR-LABEL: @loop_nested_disjoint_or(
; IR: loop:
; IR: %v0 = load i64, ptr addrspace(3) %p0
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 1024
; IR-NEXT: %v1 = load i64, ptr addrspace(3) %lds.offset
; IR: ret void
; GCN-LABEL: loop_nested_disjoint_or:
; GCN-NOT: ds_read_b64
; GCN: ds_read2{{(st64)?}}_b64
; GCN-NOT: ds_read_b64
; GCN: s_endpgm
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lo = and i32 %tid, 920
  %hi = and i32 %tid, 12288
  %wave = lshr exact i32 %hi, 8
  %full = or i32 %lo, %hi
  %wave32 = add i32 %wave, 32
  %sum0 = add i32 %wave32, %lo
  %low0 = and i32 %sum0, 1016
  %off0 = or i32 %low0, %hi
  %full32 = or i32 %full, 1056
  %sum1 = add i32 %full32, %wave
  %low1 = and i32 %sum1, 1023
  %high1 = and i32 %full32, 13312
  %off1 = or i32 %low1, %high1
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %slot = and i32 %i, 1
  %base = shl i32 %slot, 14
  %p = getelementptr i8, ptr addrspace(3) @lds, i32 %base
  %p0 = getelementptr i8, ptr addrspace(3) %p, i32 %off0
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %p1 = getelementptr i8, ptr addrspace(3) %p, i32 %off1
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  %op = getelementptr i64, ptr addrspace(1) %out, i32 %i
  store i64 %r, ptr addrspace(1) %op
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %end
end:
  ret void
}

; Explicit disjointness is sufficient even without known-zero-bit masks.
define amdgpu_kernel void @variable_disjoint_or(ptr addrspace(1) %out, i32 %x, i32 %y) {
; IR-LABEL: @variable_disjoint_or(
; IR: %lds.offset = getelementptr i8, ptr addrspace(3) %p0, i32 1024
; IR-NEXT: %v1 = load i64, ptr addrspace(3) %lds.offset
; IR: ret void
  %sum = add i32 %x, %y
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %sum
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %bits = or disjoint i32 %x, %y
  %row = add i32 %bits, 1024
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %row
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}

; Without disjointness, OR and ADD differ when the two variables overlap.
define amdgpu_kernel void @variable_overlapping_or(ptr addrspace(1) %out, i32 %x, i32 %y) {
; IR-LABEL: @variable_overlapping_or(
; IR-NOT: %lds.offset
; IR: ret void
  %sum = add i32 %x, %y
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %sum
  %v0 = load i64, ptr addrspace(3) %p0, align 8
  %bits = or i32 %x, %y
  %row = add i32 %bits, 1024
  %p1 = getelementptr i8, ptr addrspace(3) @lds, i32 %row
  %v1 = load i64, ptr addrspace(3) %p1, align 8
  %r = add i64 %v0, %v1
  store i64 %r, ptr addrspace(1) %out
  ret void
}
