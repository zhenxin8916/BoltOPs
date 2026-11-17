"""Freeze inputs once; benchmark 48 TGs x two independent M outputs, CU0 trace."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from boltops.tools.gluon_moe_gemm1 import prepare_cpu, save_kernel
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("candidate_kernel", Path(os.environ.get("KERNEL_SOURCE", str(Path(__file__).resolve().parents[4] / "boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm32xbn512xbk64.py"))))
mod = importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

ROOT = Path(os.environ.get("BM_INPUT_ROOT", str(Path.cwd() / "inputs")))
ROOT.mkdir(parents=True, exist_ok=True)
CASE = dict(M=128, E=256, N=256, K=2048, topk=8)
TOKEN_ROUTES = CASE["M"]*CASE["topk"]
BM, BN, BK, GRID, M_BLOCKS = int(os.environ.get("BENCH_BM", "32")), 512, int(os.environ.get("BENCH_BK","64")), 48, int(os.environ.get("BENCH_M_BLOCKS", "96"))

def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def freeze():
    t, *_, routing = prepare_cpu(CASE, "fp8_channel", BM, BN, 0, 3101, gemm="gemm2")
    encoded = t["Sorted"].long()
    routes = (encoded & 0xffffff) * 8 + ((encoded >> 24) & 255)
    valid = routes < TOKEN_ROUTES
    gather = routes.clamp_max(TOKEN_ROUTES - 1)
    for kind in ("fp8",):
        host = {name: value.clone() for name, value in t.items()}
        if kind == "int8":
            host["A"] = (host["A"].float() * 16).round().clamp(-127, 127).to(torch.int8)
            host["B"] = (host["B"].float() * 16).round().clamp(-127, 127).to(torch.int8)
            host["SA"] /= 16
            host["SB"] /= 16
        sorted_a = host["A"].view(torch.uint8)[gather].clone()
        sorted_a[~valid] = 0
        host["A"] = sorted_a.view(host["A"].dtype)
        host["SA"] = host["SA"][gather].clone()
        host["SA"][~valid] = 0
        golden = torch.zeros((TOKEN_ROUTES, BN), dtype=torch.bfloat16)
        mask = torch.zeros(TOKEN_ROUTES, dtype=torch.bool)
        for m in range(M_BLOCKS):
            span = slice(m * BM, (m + 1) * BM)
            rt = routes[span][valid[span]]
            expert = int(host["Experts"][m])
            a = host["A"][span].float()[valid[span]]
            value = (((a @ host["B"][expert, :BN].float().T) * host["SA"][span][valid[span]])
                     * host["SB"][expert, :BN, 0][None, :]) * host["RouteWeights"][rt, None]
            golden[rt] = value.bfloat16()
            mask[rt] = True
        hashes = {name: sha(value) for name, value in host.items()}
        hashes.update(golden=sha(golden), selected_rows=sha(mask))
        torch.save(dict(host=host, golden=golden, selected_rows=mask), ROOT / f"frozen-{kind}.pt")
        (ROOT / f"frozen-{kind}.json").write_text(json.dumps(dict(case=CASE, seed=3101, input_sha256=hashes, routing=routing), indent=2)+"\n")
        expected = Path(__file__).with_name("frozen-input-manifest.json")
        if expected.exists():
            assert hashes == json.loads(expected.read_text())["input_sha256"], "Frozen input regeneration differs from the measured inputs"
        print("FROZEN", kind, int(mask.sum()) * BN, flush=True)

def run(kind):
    frozen = torch.load(ROOT / f"frozen-{kind}.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((ROOT / f"frozen-{kind}.json").read_text())
    host = frozen["host"]
    assert all(sha(t) == manifest["input_sha256"][n] for n,t in host.items())
    gpu = {name:value.cuda() for name,value in host.items()}
    c = torch.full((TOKEN_ROUTES, 2048), float("nan"), dtype=torch.bfloat16).cuda()
    options = dict(mod.DEFAULT_LAUNCH_OPTIONS, N=2048, K=256, top_k=8,
                   BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                   CK_SORTING=True, N_TILES_LIMIT=1)
    rows = host["Sorted"].numel()
    arguments = (gpu["A"],gpu["B"],gpu["SA"],gpu["SB"],c,gpu["SortedWeights"],gpu["Sorted"],gpu["Experts"],rows,TOKEN_ROUTES,M_BLOCKS*4)
    # TOTAL_TILES uses the full-N coordinate system; only N0 is emitted here.
    compiled = mod.gemm_bottom_w8a8_channel_wasp_persistent.warmup(*arguments, grid=(GRID,), **options)
    Path("gemm").mkdir(exist_ok=True)
    import os
    os.chdir("gemm"); save_kernel(compiled); os.chdir("..")
    launch = dict(case=CASE, kind=kind, grid=GRID, bm=BM,bn=BN,bk=BK,
                  cu_count=48, tiles_per_tg=2,total_tiles=96,output_elements_per_tg=BM*BN*(M_BLOCKS//GRID),
                  kernel_total_tiles_argument=M_BLOCKS*4,N_TILES_LIMIT=1,
                  input_sha256=manifest["input_sha256"],options=options,
                  source_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
                  gpu_buffers={n:dict(address=v.data_ptr(),bytes=v.numel()*v.element_size()) for n,v in dict(gpu,C=c).items()})
    Path("launch.json").write_text(json.dumps(launch,indent=2)+"\n")
    if os.environ.get("COMPILE_ONLY") == "1":
        print("COMPILE_ONLY",compiled.metadata,flush=True);return
    mod.gemm_bottom_w8a8_channel_wasp_persistent[(GRID,)](*arguments, **options)
    actual = c.cpu()
    mask = frozen["selected_rows"]
    got, expected = actual[mask,:BN], frozen["golden"][mask]
    errors = int((~torch.isclose(got.float(),expected.float(),rtol=.02,atol=.002)).sum())
    untouched = bool(torch.isnan(actual[:,BN:]).all() and torch.isnan(actual[~mask,:BN]).all())
    result = dict(verified=True,mismatches=errors,checked_elements=got.numel(),untouched_tail=untouched,
                  output_sha256=sha(got),shared=compiled.metadata.shared,
                  scope="bounded_two_output_tiles_per_tg",tiles_per_tg=2)
    Path("result.json").write_text(json.dumps(result,indent=2)+"\n")
    assert errors == 0 and untouched, result
    print("PASS",result,flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");p.add_argument("--kind",choices=("fp8","int8"));a=p.parse_args()
    if a.freeze:freeze()
    else:run(a.kind)
