import os,json
from pathlib import Path
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import hcu
from boltops.tools.gluon_moe_gemm1 import save_kernel

@gluon.jit
def pure_mmac(A, B, C, ROUNDS, REPS: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16,16,32], 3, [2,2], tiles_per_warp=[1,2])
    out: gl.constexpr = hcu.MmacInterleave2Layout(mma)
    ld: gl.constexpr = gl.BlockedLayout([1,4],[4,16],[4,1],[1,0])
    ar=gl.arange(0,64,gl.SliceLayout(1,ld));ak=gl.arange(0,32,gl.SliceLayout(0,ld))
    br=gl.arange(0,32,gl.SliceLayout(1,ld));bn=gl.arange(0,128,gl.SliceLayout(0,ld))
    a = gl.convert_layout(gl.load(A+ar[:,None]*32+ak[None,:]),gl.DotOperandLayout(0,mma,8))
    b = gl.convert_layout(gl.load(B+br[:,None]*128+bn[None,:]),hcu.MmacInterleave2Layout(gl.DotOperandLayout(1,mma,8)))
    c = gl.full((64,128), 0, gl.float32, out)
    hcu.sched_barrier()
    for _ in range(ROUNDS):
        for _k in gl.static_range(REPS):
            c = gl.amd.cdna4.mfma(a,b,c)
    hcu.sched_barrier()
    row=gl.arange(0,64,gl.SliceLayout(1,out))
    col=gl.arange(0,128,gl.SliceLayout(0,out))
    gl.store(C+row[:,None]*128+col[None,:],c)

a=torch.ones((64,32),dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
b=torch.ones((32,128),dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
root=Path.cwd();c=torch.full((64,128),float('nan'),dtype=torch.float32).cuda()
(root/'gemm').mkdir(exist_ok=True);os.chdir(root/'gemm')
k=pure_mmac.warmup(a,b,c,17,128,grid=(1,),num_warps=4);save_kernel(k);os.chdir(root)
pure_mmac[(1,)](a,b,c,17,128,num_warps=4);torch.cuda.synchronize()
v=c.cpu();bad=int((v!=69632).sum());(root/'result.json').write_text(json.dumps(dict(mismatches=bad,checked=8192,expected=69632),indent=2));assert bad==0
