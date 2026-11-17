"""Reproduce the fixed M128, grid48, two-output-per-TG FP8 experiment."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import torch
from boltops.fused_moe.gluon import gemm_top_w8a8_channel_wasp_persistent_bm32xbn256xbk256 as mod

repo = Path("/zhenxin/BoltOPs")
spec = importlib.util.spec_from_file_location("data", repo / "boltops/tools/gluon_moe_gemm1.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
r = Path.cwd()
# The GEMM1 fixture takes intermediate N and creates the two projection halves.
t, _, _, _, routing = data.prepare_cpu(
    dict(M=128, E=256, N=256, K=2048, topk=8), "fp8_channel", 32, 256, 0, 3101)
assert t["B"].shape == (256, 512, 2048)
blocks, total, grid = 48, 96, 48
rows = t["Sorted"].numel()
(r / "input.json").write_text(json.dumps(dict(
    BM=32, BN=256, BK=256, blocks=blocks, total=total, grid=grid,
    gc_ghz=float(os.environ.get("GC_GHZ", "1.8")), routing=routing), indent=2))
(r / "kernel_snapshot.py").write_bytes(Path(mod.__file__).read_bytes())
compile_only = os.environ.get("COMPILE_ONLY") == "1"
gpu = t if compile_only else {name: value.cuda() for name, value in t.items()}
out = torch.full((rows, 512), float("nan"), dtype=torch.bfloat16)
counter = torch.tensor([blocks * 32], dtype=torch.int32)
if not compile_only:
    out, counter = out.cuda(), counter.cuda()
opts = dict(mod.DEFAULT_LAUNCH_OPTIONS,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256,
            TOTAL_TILES=total, USE_ADDR_OFFSET_INT64_A=False,
            USE_ADDR_OFFSET_INT64_B=False, USE_ADDR_OFFSET_INT64_C=False,
            num_tokens_post_padded_ptr=counter, PERSISTENT_GRID=grid)
args = (gpu["A"], gpu["B"], gpu["SA"], gpu["SB"], out,
        gpu["Sorted"], gpu["Experts"], 128, 256, 512, 2048, 8,
        *t["SB"].stride(), *t["SA"].stride())
kernel = mod.gemm_top_w8a8_channel_wasp_persistent
compiled = (kernel.warmup(*args, grid=(grid,), **opts) if compile_only
            else kernel[(grid,)](*args, **opts))
# Do not change process cwd while PMD's asynchronous trace writer is active.
if not compile_only:
    torch.cuda.synchronize()
artifact = r / "gemm"
artifact.mkdir(exist_ok=True)
os.chdir(artifact)
data.save_kernel(compiled)
os.chdir(r)
if compile_only:
    print("COMPILE_ONLY_PASS", flush=True)
    raise SystemExit(0)
actual = out.cpu()
errors = padding = checked = 0
for block in range(blocks):
    span = slice(block * 32, (block + 1) * 32)
    tokens = t["Sorted"][span].long() & 0xFFFFFF
    valid = tokens < 128
    expert = int(t["Experts"][block])
    reference = ((t["A"].float()[tokens[valid]] @ t["B"][expert].float().T)
                 * t["SA"][tokens[valid]] * t["SB"][expert, :, 0][None, :]).bfloat16()
    got = actual[span]
    errors += int((~torch.isclose(got[valid].float(), reference.float(), rtol=.02, atol=.002)).sum())
    padding += int((got[~valid] != 0).sum())
    checked += reference.numel()
untouched = int((~torch.isnan(actual[blocks * 32:])).sum())
result = dict(BM=32, BN=256, BK=256, blocks=blocks, total=total, grid=grid,
              errors=errors, padding_errors=padding, untouched_errors=untouched,
              checked=checked, shared=compiled.metadata.shared,
              output_prefix_sha256=hashlib.sha256(actual[:blocks*32].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
              text_sha256=hashlib.sha256(compiled.asm["amdgcn"].encode()).hexdigest())
(r / "result.json").write_text(json.dumps(result, indent=2))
print(result, flush=True)
assert errors == padding == untouched == 0
