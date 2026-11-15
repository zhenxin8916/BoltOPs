"""Frozen M4096/E256 GEMM1; 48 TGs x two BN512 outputs, CU0 trace."""
import argparse, hashlib, json, os
from pathlib import Path
import torch
from boltops.tools.gluon_moe_gemm1 import prepare_cpu, save_kernel
from boltops.fused_moe.gluon import gemm_top_w8a8_channel_wasp_persistent_bm128xbn512xbk128 as mod
ROOT=Path(__file__).resolve().parent
CASE=dict(M=4096,E=256,N=512,K=2048,topk=8)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def freeze():
    tensors,*_,routing=prepare_cpu(CASE,"fp8_channel",128,512,0,3101)
    for kind in ("fp8","int8"):
        host={n:t.clone() for n,t in tensors.items()}
        if kind=="int8":
            for name,scale in (("A","SA"),("B","SB")):
                host[name]=(host[name].float()*(127/448)).round().clamp(-127,127).to(torch.int8)
                host[scale]*=448/127
        golden=torch.zeros((48*128,1024),dtype=torch.bfloat16);valid_count=0
        for m in range(48):
            ids=host["Sorted"][m*128:(m+1)*128].long()&0xffffff
            valid=ids<4096;expert=int(host["Experts"][m]);valid_count+=int(valid.sum())*1024
            golden[m*128:(m+1)*128][valid]=((host["A"].float()[ids[valid]]@host["B"][expert].float().T)*host["SA"][ids[valid]]*host["SB"][expert,:,0][None,:]).bfloat16()
        hashes={n:sha(t) for n,t in host.items()};hashes["golden"]=sha(golden)
        torch.save(dict(host=host,golden=golden,checked_valid_elements=valid_count),ROOT/f"frozen-{kind}.pt")
        (ROOT/f"frozen-{kind}.json").write_text(json.dumps(dict(case=CASE,seed=3101,input_sha256=hashes,routing=routing),indent=2)+"\n")
        print("FROZEN",kind,valid_count,flush=True)
def run(kind,warm):
    frozen=torch.load(ROOT/f"frozen-{kind}.pt",map_location="cpu",weights_only=False)
    manifest=json.loads((ROOT/f"frozen-{kind}.json").read_text());host=frozen["host"]
    assert all(sha(t)==manifest["input_sha256"][n] for n,t in host.items())
    gpu={n:t.cuda() for n,t in host.items()}
    output=torch.full((host["Sorted"].numel(),1024),float("nan"),dtype=torch.bfloat16).cuda()
    args=(gpu["A"],gpu["B"],gpu["SA"],gpu["SB"],output,gpu["Sorted"],gpu["Experts"],4096,256,1024,2048,8,*host["SB"].stride(),*host["SA"].stride())
    opts=dict(mod.DEFAULT_LAUNCH_OPTIONS,TOTAL_TILES=96,BLOCK_SIZE_M=128,BLOCK_SIZE_N=512,BLOCK_SIZE_K=128,INSN_WARMUP=warm)
    compiled=mod.gemm_top_w8a8_channel_wasp_persistent.warmup(*args,grid=(48,),**opts)
    Path("gemm").mkdir(exist_ok=True);os.chdir("gemm");save_kernel(compiled);os.chdir("..")
    launch=dict(case=CASE,kind=kind,warmup=int(warm),grid=48,bm=128,bn=512,bk=128,actual_n=1024,actual_k=2048,cu_count=48,
                tiles_per_tg=2,total_tiles=96,output_elements_per_tg=128*1024,input_sha256=manifest["input_sha256"],options=opts,
                source_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
                gpu_buffers={n:dict(address=t.data_ptr(),bytes=t.numel()*t.element_size()) for n,t in dict(gpu,C=output).items()})
    Path("launch.json").write_text(json.dumps(launch,indent=2)+"\n")
    mod.gemm_top_w8a8_channel_wasp_persistent[(48,)](*args,**opts)
    actual=output.cpu();got=actual[:6144];gold=frozen["golden"]
    errors=int((~torch.isclose(got.float(),gold.float(),rtol=.02,atol=.002)).sum())
    untouched=bool(torch.isnan(actual[6144:]).all())
    result=dict(verified=True,mismatches=errors,checked_elements=frozen["checked_valid_elements"],checked_padded_elements=gold.numel()-frozen["checked_valid_elements"],untouched_tail=untouched,output_sha256=sha(got),shared=compiled.metadata.shared)
    Path("result.json").write_text(json.dumps(result,indent=2)+"\n")
    assert errors==0 and untouched,result
    print("PASS",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");p.add_argument("--kind",choices=("fp8","int8"));p.add_argument("--warm",action="store_true");a=p.parse_args()
    if a.freeze:freeze()
    else:run(a.kind,a.warm)
