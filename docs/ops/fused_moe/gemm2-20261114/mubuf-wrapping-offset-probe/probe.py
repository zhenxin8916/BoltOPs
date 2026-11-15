import os,json
from pathlib import Path
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import buffer_load,buffer_store
@gluon.jit
def probe(S, X, O):
    layout:gl.constexpr=gl.BlockedLayout([1],[64],[4],[0])
    n=gl.arange(0,16,layout)
    biased=gl.load(X+n).to(gl.uint32)
    normal=buffer_load(S,(n*128+64).to(gl.uint32))
    split=gl.inline_asm_elementwise(
        "s_mov_b64 s[0:1], $1\ns_mov_b32 s2, 0xfffffff8\ns_mov_b32 s3, 0x27000\nbuffer_load_dword $0, $2, s[0:3], 0 offen offset:3840\ns_waitcnt vmcnt(0)",
        constraints="=&v,s,v,~{s0},~{s1},~{s2},~{s3}",args=[S.to(gl.uint64),biased],dtype=gl.float32,is_pure=False,pack=1)
    full,tmp=gl.inline_asm_elementwise(
        "s_mov_b64 s[0:1], $2\ns_mov_b32 s2, 0xfffffff8\ns_mov_b32 s3, 0x27000\nv_add_u32 $1, 3840, $3\nbuffer_load_dword $0, $1, s[0:3], 0 offen\ns_waitcnt vmcnt(0)",
        constraints="=&v,=&v,s,v,~{s0},~{s1},~{s2},~{s3}",args=[S.to(gl.uint64),biased],dtype=(gl.float32,gl.uint32),is_pure=False,pack=1)
    buffer_store(normal,O,(n*4).to(gl.uint32))
    buffer_store(split,O,(n*4+1).to(gl.uint32))
    buffer_store(full,O,(n*4+2).to(gl.uint32))
    buffer_store(tmp.to(gl.float32),O,(n*4+3).to(gl.uint32))
s=torch.arange(1,8193,dtype=torch.float32);x=torch.arange(16,dtype=torch.int32)*512-3584;o=torch.full((16,4),-1,dtype=torch.float32)
sg,xg,og=s.cuda(),x.cuda(),o.cuda()
k=probe[(1,)](sg,xg,og,num_warps=4,enable_fp_fusion=False)
torch.cuda.synchronize();out=og.cpu();expected=s[torch.arange(16)*128+64]
for ext in ['amdgcn','llir','hsaco']:
 v=k.asm[ext];(Path.cwd()/('kernel.'+ext)).write_bytes(v if isinstance(v,bytes) else v.encode())
r=dict(columns=['ordinary buffer_load','split voffset+imm3840','full u32 sum then imm0','wrapped byte offset'],actual=out.tolist(),expected=expected.tolist(),normal_ok=torch.equal(out[:,0],expected),full_add_ok=torch.equal(out[:,2],expected),split_bad_indices=torch.where(out[:,1]!=expected)[0].tolist())
Path('result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);assert r['normal_ok'] and r['full_add_ok']
