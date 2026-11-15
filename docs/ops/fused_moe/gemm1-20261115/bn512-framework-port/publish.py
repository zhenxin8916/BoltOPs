"""Publish verified BN512 GEMM1 results without changing other CSV rows."""
import gzip, hashlib, json, shutil, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
from boltops.tools.gluon_moe_experiment_csv import upsert_record
from analyze import analyze
ROOT = Path(__file__).resolve().parent
REPO = Path("/zhenxin/BoltOPs")
DEST = REPO / "docs/ops/fused_moe/gemm1-20261115/bn512-framework-port"
SOURCE = REPO / "boltops/fused_moe/gluon/gemm_top_w8a8_channel_wasp_persistent_bm128xbn512xbk128.py"
def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    triton = Path("/zhenxin/worktrees/triton-feature-20261102")
    llvm = Path("/zhenxin/llvm-project")
    assert git(triton, "rev-parse", "HEAD") == "5b24821ee57eeb7560074cc7979ce7b59604079f"
    assert not git(triton, "status", "--porcelain")
    toolchain = dict(triton_commit=git(triton, "rev-parse", "HEAD"), triton_changes="无本地改动",
        llvm_project_commit=git(llvm, "rev-parse", "HEAD"),
        llvm_changes="提交修复 buffer 32 位 offset 回绕并归一 LDS 地址、合并 DS 读取；本地补丁禁止 load 下沉跨 barrier，避免槽位复用竞态；启用 hcu-control-only-ebarrier-sync，避免控制同步排空 LDS 读取。")
    DEST.mkdir(parents=True, exist_ok=True)
    summaries=[]
    addresses={}
    for kind in ("fp8", "int8"):
        manifest=json.loads((ROOT/f"frozen-{kind}.json").read_text())
        outputs=set(); binaries=set(); pointers=[]
        for gc in (2200,1800,1500):
            directory=ROOT/f"{kind}-gc{gc}"
            metrics_path=directory/"metrics.json"
            metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else analyze(directory)
            assert metrics["hardware_wait_rules_passed"] and len(metrics["active_wave_counts"])==8
            assert set(metrics["active_wave_counts"].values())=={4096}
            launch=json.loads((directory/"launch.json").read_text())
            result=json.loads((directory/"result.json").read_text())
            assert launch["input_sha256"] == manifest["input_sha256"]
            assert launch["source_sha256"] == digest(SOURCE)
            assert (launch["tiles_per_tg"],launch["total_tiles"],launch["grid"],launch["warmup"]) == (2,96,48,0)
            outputs.add(result["output_sha256"])
            binaries.add(digest(directory/"gemm/kernel.hsaco"))
            pointers.append(launch["gpu_buffers"])
            row=dict(arch="gfx946", quant_type="int8_w8a8_channel" if kind=="int8" else "f8_w8a8_channel",
                indtype="torch.bfloat16",token=4096,inter_dim=512,model_dim=2048,expert=256,topk=8,
                sol_type="triton",sol_id="triton:gluon_moe_gemm1",bm=128,bn=512,bk=128,
                actual_n=1024,actual_k=2048,cu_count=48,tiles_per_tg=2,total_tiles=96,
                output_elements_per_tg=128*1024,time_scope="full_dispatch",
                work_scope="bounded_equal_padded_output_work",warmup=0,**toolchain,**result)
            row.update({k:metrics[k] for k in ("time_us","wave_span_time_us","mmac_total_pct","mmac_scope","gc_ghz","effective_gc_ghz","clock_period_ps")})
            row.update(kernel_sha256=digest(SOURCE),input_sha256=manifest["input_sha256"],
                evidence=f"docs/ops/fused_moe/gemm1-20261115/bn512-framework-port/{directory.name}")
            out=DEST/directory.name;out.mkdir(exist_ok=True)
            for name, data in (("metrics.json",metrics),("launch.json",launch),("result.json",result),("record.json",row)):
                (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
            shutil.copyfile(directory/"gemm/compiled.json",out/"compiled.json")
            upsert_record("top",row)
            summaries.append(dict(kind=kind,gc_ghz=gc/1000,time_us=metrics["time_us"],
                mmac_total_pct=metrics["mmac_total_pct"],hot_loop_mmac_pct=metrics["hot_loop_mmac_pct"],
                checked_elements=result["checked_elements"],mismatches=0))
        assert len(outputs)==len(binaries)==1,(kind,outputs,binaries)
        addresses[kind]=all(p==pointers[0] for p in pointers)
    warm=ROOT/"fp8-warm-gc1800"
    warm_metrics=json.loads((warm/"metrics.json").read_text()) if (warm/"metrics.json").exists() else analyze(warm)
    warm_launch=json.loads((warm/"launch.json").read_text())
    warm_result=json.loads((warm/"result.json").read_text())
    cold_result=json.loads((ROOT/"fp8-gc1800/result.json").read_text())
    assert warm_launch["warmup"]==1 and warm_launch["source_sha256"]==digest(SOURCE)
    assert warm_launch["input_sha256"]==json.loads((ROOT/"frozen-fp8.json").read_text())["input_sha256"]
    assert warm_result["output_sha256"]==cold_result["output_sha256"]
    warm_out=DEST/warm.name;warm_out.mkdir(exist_ok=True)
    for name in ("metrics.json","launch.json","result.json"):
        shutil.copyfile(warm/name,warm_out/name)
    tests=ET.parse(ROOT/"check/equivalence.xml")
    reports=[json.loads(p.attrib["value"]) for p in tests.findall(".//property[@name='equivalence']")]
    assert len(reports)==6 and all(r["differing_elements"]==0 for r in reports)
    (DEST/"equivalence.json").write_text(json.dumps(reports,indent=2)+"\n")
    assert "216 passed, 6 skipped" in (ROOT/"check/combined-cpu-tests.log").read_text()
    (DEST/"validation.json").write_text(json.dumps(dict(cpu_passed=216,opt_in_skipped=6,
        model_equivalence_passed=6,model_elements_per_case=393216,model_differing_elements=0,
        cpu_files=["test_gluon_fused_experts.py","test_channel_gemm1_bn512_equivalence.py",
                   "test_channel_gemm2_dispatch.py","test_channel_gemm2_bn512_dispatch.py"]),indent=2)+"\n")
    for name in ("run.py","launch.py","analyze.py","publish.py","perf_dispatch_span.py","frozen-fp8.json","frozen-int8.json"):
        if (ROOT/name).resolve()!=(DEST/name).resolve():shutil.copyfile(ROOT/name,DEST/name)
    source_dir="experiments/moe_channel_ab_shared8/persistent_bn512/"
    originals={}
    for name in ("kernel_bn512_retained_a_native_fragment.py","bn512_retained_a_producers.py","bn512_epilogue_native_fragment.py","bn512_epilogue.py"):
        data=subprocess.check_output(["git","-C",str(REPO),"show",f"5f9c36d:{source_dir}{name}"])
        originals[name]=hashlib.sha256(data).hexdigest()
    provenance=dict(**toolchain,kernel_sha256=digest(SOURCE),boltops_parent=git(REPO,"rev-parse","HEAD"),
        original_commit=git(REPO,"rev-parse","5f9c36d"),original_sources=originals,
        clang_sha256=digest(Path("/zhenxin/llvm-project/build-dtk-llvm-6.3.3/bin/clang-18")),
        gpu_addresses_equal_across_clocks=addresses,raw_root=str(ROOT).replace("/zhenxin/","/home/zhenx/",1))
    (DEST/"provenance.json").write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+"\n")
    (DEST/"llvm-local.patch.gz").write_bytes(gzip.compress((git(llvm,"diff")+"\n").encode(),mtime=0))
    (DEST/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    table="\n".join(f"| {r['kind'].upper()} | {r['gc_ghz']:.1f} | {r['time_us']:.6f} | {r['mmac_total_pct']:.4f} | {r['hot_loop_mmac_pct']:.4f} |" for r in summaries)
    (DEST/"README.md").write_text("""# BN512 channel GEMM1 production port

Single kernel file: `gemm_top_w8a8_channel_wasp_persistent_bm128xbn512xbk128.py`.
Source algorithm: commit 5f9c36d, four original files listed with hashes in provenance.json.
P0 publishes both B halves through a three-slot FIFO; P1 publishes A through two
slots. Consumers retain each K256 A pair across two N128 panels. Native fragment
stores preserve wave ownership. A2*16KiB + B3*32KiB = 128KiB LDS.

Production additions: CK packed routes; INT8 exact INT32 accumulation; BF16/FP16
output; non-unit channel-scale strides; independent 64-bit A/B/C constants;
caller CU budget, device work count and complete N traversal. The CPU query
selects BN512 for complete N512/K512 cycles; other shapes retain existing
BN128/256 or ordinary Triton selection. No compiler source change is needed.
Warmup defaults to false. No spill was reported by the compiler.

## Normalized three-clock measurements

M4096, E256, topk8, actual GEMM1 N1024/K2048, BM128/BN512/BK128.
48 CUs, one persistent TG each, two outputs per TG: 96 output tiles and
128*1024 padded elements per TG, matching BN256*4 and BN128*8 CSV work.
TOTAL_TILES=96 bounds this diagnostic; production traverses all output tiles.
Each type reuses one frozen seed3101 input/routing/reference at all clocks.

| Type | GC (GHz) | Full dispatch (us) | CU0 lifetime MMAC (%) | Hot MMAC (%) |
|---|---:|---:|---:|---:|
"""+table+f"""

CU0 efficiency averages its four SIMD active MMAC interval unions over its own
first WAVE_START through last s_endpgm plus one cycle, including startup/tail.
It does not divide by the full 48-CU dispatch window. Full dispatch time comes
from the HYGPERF XCD time section. All 48 CUs complete one TG. Eight CU0 consumer
waves execute 4096 active FP8/INT8 MMACs each. BF16 accumulator clears and EXEC=0
warmup instructions are excluded. Hot efficiency averages each output/SIMD's
first through last active quantized MMAC span. Every wave's first abarrier wait
is nonzero; consecutive waits use different IDs; all IDs remain within 0..15.

All six runs match the CPU reference (rtol .02, atol .002): 4,314,112 valid
and 1,977,344 padded elements checked; unselected output rows remain NaN.
Per-type input/output hashes and HSACO are identical across clocks. Address
comparisons, compiler register quotas, exact toolchain and local LLVM patch
are archived. Default clock is 1.8GHz; all clock overrides are process-local.

Archived 5f9c36d reported warmup=true: 59.425280us / hot MMAC84.90% on its older
toolchain. Current warmup=true rerun at1.8GHz is {warm_metrics['time_us']:.6f}us /
hot MMAC{warm_metrics['hot_loop_mmac_pct']:.4f}%. It is retained as a separate
comparison, not substituted for the default warmup=false CSV measurements.

## Correctness and limits

Six independent ordinary-Triton/CPU comparisons cover E4M3/E5M2/mixed FP8,
INT8, GROUP_SIZE_M=2, BF16 normal
addresses, FP16 forced64-bit A/B/C, scale strides, padded/invalid experts and
six persistent outputs across M and N. Each compares393216 elements exactly;
zero differing elements. Combined CPU/framework regression:216 passed, six
opt-in model cases skipped there and separately run successfully. See
equivalence.json, validation.json and the opt-in pytest. Forced
64-bit branches were tested on small allocations; no physical >4GiB input
was allocated. This BN512 algorithm requires K>=512, K%512=0, N%512=0.

Narrow A preserves native buffer-to-LDS copies. Wide A uses two K64 packed
int32 global-load/LDS-store chunks per K128 slot, with a plain-row shared
layout. This limits producer temporaries while preserving the two A slots.
B uses MLS in both paths. Query rejection preserves ordinary Triton fallback.

## Reproduction

Run in the paired container environment specified by launch.py:

```bash
python run.py --freeze  # once; retain the frozen tensors for all comparisons
python launch.py fp8
python launch.py int8
python launch.py fp8 --warm
mkdir -p check
RUN_GLUON_GEMM1_BN512_EQUIVALENCE=1 python -m pytest /zhenxin/BoltOPs/test/fused_moe/test_channel_gemm1_bn512_equivalence.py --junitxml=check/equivalence.xml
python -m pytest /zhenxin/BoltOPs/test/fused_moe/test_gluon_fused_experts.py /zhenxin/BoltOPs/test/fused_moe/test_channel_gemm1_bn512_equivalence.py /zhenxin/BoltOPs/test/fused_moe/test_channel_gemm2_dispatch.py /zhenxin/BoltOPs/test/fused_moe/test_channel_gemm2_bn512_dispatch.py -q > check/combined-cpu-tests.log
python publish.py
```

Opt-in numerical comparison:
`RUN_GLUON_GEMM1_BN512_EQUIVALENCE=1 python -m pytest test/fused_moe/test_channel_gemm1_bn512_equivalence.py`.
Only compact evidence is tracked; raw CU0 traces and frozen tensor payloads
remain at raw_root in provenance.json. CSVs live beside the kernel: top_fp8.csv
and top_int8.csv with the full tuned_experiment_fmoe_gluon_w8a8channel_ prefix.
""")
    print(json.dumps(summaries,indent=2))
if __name__=="__main__":main()
