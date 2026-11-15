; RUN: llc -march=amdgcn -mcpu=gfx946 -O3 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=buffer_wbinvl1
;
; A conditional mailbox store must not sink the LDS read past release of its
; EMPTY credit. Another wave can overwrite that physical slot at release.
; hasSideEffects alone on the abarrier instruction does not protect the load
; in MachineSink: target sinking legality must preserve this ordering.
;
; CHECK-LABEL: release_credit:
; CHECK: ds_read_b32
; CHECK: s_abarrier_arrive
; CHECK: s_cbranch
; CHECK-NOT: ds_read
; CHECK: s_endpgm

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @release_credit(ptr addrspace(3) %shared, ptr addrspace(1) %out, i1 %write) {
entry:
  %value = load i32, ptr addrspace(3) %shared, align 4
  %credit = call i32 (i32, ...) @llvm.hcu.s.abarrier.arrive(i32 11, i32 1)
  br i1 %write, label %store, label %exit
store:
  store i32 %value, ptr addrspace(1) %out, align 4
  br label %exit
exit:
  ret void
}

declare i32 @llvm.hcu.s.abarrier.arrive(i32, ...)
