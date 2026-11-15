"""Publish verified BN256/4-output GEMM2 measurements; preserve other tiles."""
import csv, gzip, hashlib, json, re, shutil, subprocess
from pathlib import Path
from boltops.tools.gluon_moe_experiment_csv import upsert_record, FIELDS
from boltops.tools.fused_moe_triton_tune.performance_csv import _atomic_write, csv_write_lock
from analyze import analyze

ROOT=Path(__file__).resolve().parent
REPO=Path("/zhenxin/BoltOPs")
DEST=REPO/"docs/ops/fused_moe/gemm2-20261115/bn256-four-outputs-three-clocks"
CSV_DIR=REPO/"boltops/fused_moe/gluon"

def git(repo,*args):return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    triton=Path("/zhenxin/worktrees/triton-feature-20261102")
    llvm=Path("/zhenxin/llvm-project")
    assert git(triton,"rev-parse","HEAD")=="5b24821ee57eeb7560074cc7979ce7b59604079f"
    assert not git(triton,"status","--porcelain")
    provenance=dict(triton_commit=git(triton,"rev-parse","HEAD"),triton_changes="无本地改动",
        llvm_project_commit=git(llvm,"rev-parse","HEAD"),
        llvm_changes="提交修复 buffer 32 位 offset 回绕并归一 LDS 地址、合并 DS 读取；本地补丁禁止 load 下沉跨 barrier，避免槽位复用竞态；启用 hcu-control-only-ebarrier-sync，避免控制同步排空 LDS 读取。")
    source=REPO/"boltops/fused_moe/gluon/gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn256xbk128.py"
    source_sha=digest(source)
    summaries=[];records=[]
    for kind in ("fp8","int8"):
        outputs=set();binaries=set();addresses=[]
        manifest=json.loads((ROOT/f"frozen-{kind}.json").read_text())
        for gc in (2200,1800,1500):
            directory=ROOT/f"{kind}-gc{gc}"
            metrics=analyze(directory)
            launch=json.loads((directory/"launch.json").read_text())
            result=json.loads((directory/"result.json").read_text())
            assert launch["input_sha256"]==manifest["input_sha256"]
            assert launch["source_sha256"]==source_sha
            assert (launch["tiles_per_tg"],launch["total_tiles"],launch["grid"])==(4,192,48)
            outputs.add(result["output_sha256"]);binaries.add(digest(directory/"gemm/kernel.hsaco"))
            addresses.append(launch["gpu_buffers"])
            row=dict(arch="gfx946",quant_type="int8_w8a8_channel" if kind=="int8" else "f8_w8a8_channel",
                indtype="torch.bfloat16",token=4096,inter_dim=256,model_dim=2048,expert=256,topk=8,
                sol_type="triton",sol_id="triton:gluon_moe_gemm2",bm=128,bn=256,bk=128,
                actual_n=2048,actual_k=256,cu_count=48,tiles_per_tg=4,total_tiles=192,
                output_elements_per_tg=128*1024,time_scope="full_dispatch",
                work_scope="bounded_equal_padded_output_work",warmup=0,
                **provenance)
            row.update(result)
            row.update({k:metrics[k] for k in ("time_us","wave_span_time_us","mmac_total_pct","mmac_scope","gc_ghz","effective_gc_ghz")})
            row.update(clock_period_ps=metrics["clock_period_ps"],kernel_sha256=source_sha,input_sha256=manifest["input_sha256"],
                       evidence=f"docs/ops/fused_moe/gemm2-20261115/bn256-four-outputs-three-clocks/{kind}-gc{gc}")
            records.append((directory,row,metrics,launch,result))
            summaries.append(dict(kind=kind,gc_ghz=gc/1000,bm=128,bn=256,bk=128,time_us=metrics["time_us"],mmac_total_pct=metrics["mmac_total_pct"],cu0_lifetime_us=metrics["cu0_lifetime_us"],mmac_scope=metrics["mmac_scope"],mismatches=0,checked_elements=result["checked_elements"]))
        assert len(outputs)==1 and len(binaries)==1,(kind,outputs,binaries)
        provenance[f"{kind}_gpu_addresses_equal_across_clocks"]=all(a==addresses[0] for a in addresses)
    # Archive the previous BN256 row instead of mixing an old toolchain/result
    # into the current table. Preserve all other geometries, including BN512.
    DEST.mkdir(exist_ok=True)
    bottom=CSV_DIR/"tuned_experiment_fmoe_gluon_w8a8channel_bottom.csv"
    archive=DEST/"previous-bottom.csv"
    if not archive.exists():shutil.copyfile(bottom,archive)
    with csv_write_lock(bottom):
        with bottom.open(newline="") as stream:rows=list(csv.DictReader(stream))
        keep=[row for row in rows if not (row["arch"]=="gfx946" and row["token"]=="4096" and row["inter_dim"]=="256" and row["model_dim"]=="2048" and row["expert"]=="256" and row["topk"]=="8" and row["bm"]=="128" and row["bn"]=="256" and row["bk"]=="128" and row["sol_id"]=="triton:gluon_moe_gemm2")]
        def write(stream):
            writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator="\n");writer.writeheader();writer.writerows(keep)
        _atomic_write(bottom,write)
    for directory,row,metrics,launch,result in records:
        out=DEST/directory.name;out.mkdir(exist_ok=True)
        for name,data in (("record.json",row),("metrics.json",metrics),("launch.json",launch),("result.json",result)):
            (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
        shutil.copyfile(directory/"gemm/compiled.json",out/"compiled.json")
        upsert_record("bottom",row)
    for name in ("run.py","launch.py","analyze.py","publish.py","perf_dispatch_span.py","frozen-fp8.json","frozen-int8.json"):
        if (ROOT/name).resolve() != (DEST/name).resolve():
            shutil.copyfile(ROOT/name,DEST/name)
    provenance.update(kernel_sha256=source_sha,boltops_commit=git(REPO,"rev-parse","HEAD"),
        raw_root=str(ROOT).replace("/zhenxin/", "/home/zhenx/", 1),
        clang_wrapper="/zhenxin/temp/boltops-shared-ring-port-20261114/optimized-perf/clang-control-sync",
        clang_sha256=digest(Path("/zhenxin/llvm-project/build-dtk-llvm-6.3.3/bin/clang-18")))
    (DEST/"provenance.json").write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+"\n")
    (DEST/"llvm-local.patch.gz").write_bytes(gzip.compress((git(llvm,"diff")+"\n").encode(),mtime=0))
    (DEST/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    table = "\n".join(f"| {x['kind'].upper()} | {x['gc_ghz']:.1f} | {x['time_us']:.6f} | {x['mmac_total_pct']:.6f} |" for x in summaries)
    (DEST/"README.md").write_text("""# BN256 GEMM2: four outputs per TG, three clocks

Current kernel: `gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn256xbk128.py`.
M4096, E256, topk8; actual GEMM2 N2048/K256; BM128/BN256/BK128.
48 CUs each run one persistent TG and four independent M output tiles: 192
BN256 tiles, or 128*1024 padded output elements per TG. This matches the
experiment CSV normalization. N_TILES_LIMIT=1 is an explicit diagnostic;
TOTAL_TILES=1536 uses the kernel's full-N coordinate system. Production still
traverses all N. These are bounded GEMM2 experiments, not full-MoE timings.

| Type | GC (GHz) | Full dispatch (us) | CU0 whole-lifetime MMAC (%) |
|---|---:|---:|---:|
"""+table+"""

Each run checked 4,330,752 valid output elements with zero mismatches; unselected
rows/columns remained NaN. CPU reference tolerances: rtol .02, atol .002. Inputs,
CK routing and reference outputs were frozen once per type (seed3101). Across
clocks, each type has identical input/output hashes and HSACO. GPU addresses
are recorded per run and compared in provenance.json.

Only CU0 is traced. Efficiency is the four-SIMD average of active 8-cycle MMAC
interval unions, divided by CU0's first WAVE_START through its last s_endpgm
plus one cycle. This includes CU0 startup and tail; the denominator is not the
full 48-CU dispatch window. Dispatch time is read from the HYGPERF XCD time
section. All 48 CUs complete one TG and exactly 4096 MMAC instructions each;
CU0 has eight consumer waves with 512 active MMACs apiece. The interval union
equals instruction_count*8 cycles. FP8 and INT8 use their own opcode families.

CSV locations under boltops/fused_moe/gluon:
- tuned_experiment_fmoe_gluon_w8a8channel_bottom.csv (current FP8 BN256 row)
- tuned_experiment_fmoe_gluon_w8a8channel_bottom_int8.csv

The old BN256 row is archived in previous-bottom.csv; other geometries are
preserved. BN512 measurements were not changed by this task. Toolchain commit
IDs and Chinese change descriptions are included in each CSV. Triton 5b24821
has no local changes. Provenance records LLVM's local load-sinking patch and
clang's control-only ebarrier flag. No global frequency setting was modified.
Default measurement frequency remains 1.8 GHz; other clocks are per-process.

Reproduce inside the paired environment, keeping this directory writable:

```bash
python run.py --freeze
python launch.py fp8
python launch.py int8
python publish.py
```

launch.py specifies the exact Python/Triton/LLVM paths, 48-CU model and CU0-only
trace flags. Frozen tensor payloads and raw traces remain in the raw_root from
provenance.json; only scripts, manifests, validation and timing records are
tracked here. Publishing validates the numerical results, work normalization,
clock periods, one-TG-per-CU counts, input hashes and MMAC denominators first.
""")
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=="__main__":main()
