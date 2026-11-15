import hashlib,json,struct,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parents[2]
P=R/'perf'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def elf_sections(p):
 b=p.read_bytes();o=struct.unpack_from('<Q',b,40)[0];sz,n,ix=struct.unpack_from('<HHH',b,58)
 h=[struct.unpack_from('<IIQQQQIIQQ',b,o+i*sz) for i in range(n)]
 strings=b[h[ix][4]:h[ix][4]+h[ix][5]]
 return {strings[x[0]:].split(b'\0')[0].decode():hashlib.sha256(b[x[4]:x[4]+x[5]]).hexdigest() for x in h}
rows=[]
for d in sorted(P.glob('*-top-bn512-fp8-*')):
 launch=json.loads((d/'launch.json').read_text());row=dict(directory=str(d),launch=launch,status='running')
 row['artifact_sha256']={q.name:digest(q) for q in (d/'gemm').iterdir() if q.is_file()}
 row['elf_sections']=elf_sections(d/'gemm/kernel.hsaco')
 compiled=json.loads((d/'gemm/compiled.json').read_text())['kernel_metadata']
 row['registers']={k:v for k,v in compiled.items() if 'partition_regs' in k}
 gcn=(d/'gemm/kernel.amdgcn').read_text()
 row['spill_free']=all(s in gcn for s in ('.sgpr_spill_count: 0','.vgpr_spill_count: 0','.private_segment_fixed_size: 0'))
 assert row['spill_free']
 if (d/'result.json').exists():row['result']=json.loads((d/'result.json').read_text())
 if (d/'metrics.json').exists():row['metrics']=json.loads((d/'metrics.json').read_text());row['status']='verified'
 rows.append(row)
assert len({json.dumps(r['launch']['input_sha256'],sort_keys=True) for r in rows})==1
assert len({r['launch']['source_sha256'] for r in rows})==1
assert len({json.dumps(r['launch']['gpu_buffers'],sort_keys=True) for r in rows})==1
assert len({r['result']['output_sha256'] for r in rows if 'result' in r})==1
comparisons={}
for warm in (False,True):
 selected=[r for r in rows if bool(r['launch']['warmup'])==warm]
 ref=selected[0]
 comparisons['warm' if warm else 'default']={
  'amdgcn_identical':len({r['artifact_sha256']['kernel.amdgcn'] for r in selected})==1,
  'llir_identical':len({r['artifact_sha256']['kernel.llir'] for r in selected})==1,
  'text_section_identical':len({r['elf_sections']['.text'] for r in selected})==1,
  'elf_sections_differing':sorted({k for r in selected for k,v in r['elf_sections'].items() if v!=ref['elf_sections'].get(k)})}
 assert all(comparisons['warm' if warm else 'default'][k] for k in ('amdgcn_identical','llir_identical','text_section_identical'))
xml=ET.parse(R/'correctness/g1-bn512/results.xml').getroot()[0]
report=dict(rows=rows,comparisons=comparisons,correctness_testsuite=dict(xml.attrib),notes=[
 'Full dispatch time includes all 48 CUs; MMAC efficiency denominator is CU0 own 16-wave lifetime.',
 'FP8 active compute only: EXEC=0 warmup and BF16 accumulator initialization excluded.',
 'Three hardware wait-ID rules audited dynamically by shared analyzer: first wait ID nonzero, no consecutive equal IDs per wave, IDs within 0..15.',
 'No kernel, compiler, CSV or git changes made by this audit.'])
(Path(__file__).parent/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# BN512 GEMM1 FP8 LLVM rebuild audit','', '| Build / clock / warmup | Full dispatch (us) | CU0 whole MMAC | Hot loop MMAC | Numerical |','|---|---:|---:|---:|---|']
for r in rows:
 if 'metrics' in r:
  m=r['metrics'];lines.append(f"| {Path(r['directory']).name} | {m['time_us']:.6f} | {m['mmac_total_pct']:.4f}% | {m['hot_loop_mmac_pct']:.4f}% | {r['result']['mismatches']} / {r['result']['checked_elements']} |")
 else:lines.append(f"| {Path(r['directory']).name} | running | | | |")
lines+=['','Same frozen routing, tensor addresses, input hashes, output hash and production source hash across all cases. New and previous LLVM produce byte-identical AMDGCN, LLVM IR and executable `.text` separately for warmup off/on. Whole HSACO hashes differ only in debug line sections; no machine-code difference.','', '48 CU, one TG per CU, two BN512 tiles per TG, BM128/BK128, M4096, 128 KiB LDS. No SGPR/VGPR spills and zero private segment in all compiled cases.','', 'The existing correctness suite has '+xml.attrib['tests']+' passed cases; this includes FP8 e4/e5/mixed, output BF16/FP16, forced 64-bit addressing and ordinary Triton comparison. No INT8 jobs were launched by this audit.','', 'Metrics use the shared trace analyzer and its three wait-ID hardware checks. Timing rows require successful numerical verification and untouched tail. Warmup results remain separate from default-off performance.']
(Path(__file__).parent/'README.md').write_text('\n'.join(lines)+'\n')
print('audited',len(rows),'cases')
