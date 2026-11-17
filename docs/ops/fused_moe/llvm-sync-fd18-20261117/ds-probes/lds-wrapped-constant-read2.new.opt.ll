; ModuleID = '/zhenxin/temp/llvm-sync-fd18-20261117/probes/lds-wrapped-constant-read2.ll'
source_filename = "/zhenxin/temp/llvm-sync-fd18-20261117/probes/lds-wrapped-constant-read2.ll"
target datalayout = "e-p:64:64-p1:64:64-p3:32:32-i64:64-n32:64"
target triple = "amdgcn-amd-amdhsa"

@lds = internal addrspace(3) global [65536 x i8] undef, align 65536

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.x() #0

define amdgpu_kernel void @no_wrap_field(ptr addrspace(1) %out) #1 {
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
  store i64 %r, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_kernel void @no_wrap_lowmask(ptr addrspace(1) %out) #1 {
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
  store i64 %r, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_kernel void @may_wrap_field(ptr addrspace(1) %out) #1 {
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
  store i64 %r, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_kernel void @mask_has_hole(ptr addrspace(1) %out) #1 {
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
  store i64 %r, ptr addrspace(1) %out, align 8
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="gfx946" }
attributes #1 = { "target-cpu"="gfx946" }
