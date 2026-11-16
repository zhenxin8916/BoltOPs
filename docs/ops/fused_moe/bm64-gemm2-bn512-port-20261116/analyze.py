"""Audit numerical, barrier, register and three-frequency PMD evidence."""
import ast, csv, hashlib, json, re, subprocess, sys
from pathlib import Path
repo = Path(__file__).resolve().parents[4]
old = repo/'docs/ops/fused_moe/gemm2-bm64-bn512-20261115'
sys.path.insert(0, str(old))
from perf_dispatch_span import read_dispatch_span
root = Path('/zhenxin/temp/bm64-gemm2-bn512-port-20261116')
rows=[]
for case in sorted(root.glob('*-gc*')):
    result=json.loads((case/'result.json').read_text())
    assert result['mismatches']==0 and result['untouched_tail']
    for audit in ('live-barriers/audit_live_barriers.py', 'full-k256-b-slot/audit_full_k256_b_phases.py', 'audit_consumer_no_vmem.py'):
        subprocess.run([sys.executable,str(old/audit),str(case)],check=True)
    stats=(case/'m5out/0/0/stats.txt').read_text()
    def stat(name): return float(re.search(r'^'+re.escape(name)+r'\s+(\S+)',stats,re.M)[1])
    asm=(case/'gemm/kernel.amdgcn').read_text()
    counts={name:int(re.search(r'\.'+name+r':\s+(\d+)',asm)[1]) for name in ('sgpr_spill_count','vgpr_spill_count','private_segment_fixed_size')}
    assert all(v==0 for v in counts.values()) and not re.search(r'^\s*(?:scratch_|buffer_\w+[^\n]*\sscratch)',asm,re.M)
    compiled=json.loads((case/'gemm/compiled.json').read_text())['kernel_metadata']
    span=read_dispatch_span(case)
    cu_cycles=stat('system.shaders.CUs00.totalCycles')
    mmac=stat('system.shaders.CUs00.numInstrTypeExecuted::MMOP')
    trace=(case/'m5out/0/0/itrace_cu0_se0_xcd0.txt').read_text()
    period=int(re.search(r'GC_CLOCK (\d+)',trace)[1])
    starts=[int(t) for t in re.findall(r'^(\d+) [^\n]*: WAVE_START\[',trace,re.M)]
    ends=[int(t) for t in re.findall(r'^(\d+) [^\n]*: s_endpgm',trace,re.M)]
    assert (max(ends)-min(starts))/period == cu_cycles
    row=dict(kind=case.name.split('-')[0],M=1024,BM=64,BN=512,BK=64,grid=48,tiles_per_tg=2,gc_mhz=int(case.name.split('gc')[1]),time_us=span['duration_us'],cu0_mmac_percent=100*mmac*8/(4*(cu_cycles+1)),cu0_total_cycles=cu_cycles+1,cu0_stats_total_cycles=cu_cycles,cu0_denominator='first WAVE_START through last s_endpgm plus one cycle',E=256,checked=result['checked_elements'],shared_bytes=compiled['shared'],spill=counts,actual_partition_vgprs=compiled['wasp_partition_regs_actual'],allocated_partition_vgprs=compiled['wasp_partition_regs'],source_sha256=hashlib.sha256((case/'gemm/kernel.source').read_bytes()).hexdigest(),perf=span['source_perf'])
    (case/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')
    rows.append(row)
Path(__file__).with_name('results.json').write_text(json.dumps(rows,indent=2)+'\n')
fields=['kind','M','BM','BN','BK','grid','tiles_per_tg','gc_mhz','time_us','cu0_mmac_percent']
with Path(__file__).with_name('results.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
    for row in rows: writer.writerow(dict(row,cu0_mmac_percent=f"{row['cu0_mmac_percent']:.6f}%"))
print(json.dumps(rows,indent=2))
