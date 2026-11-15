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
