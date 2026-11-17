import json
from pathlib import Path
r=Path('/zhenxin/BoltOPs/docs/ops/fused_moe/llvm-sync-fd18-20261117')
rows=[]
def add(name,old,new,details,text=True,ds=True):
 a,b=old['time_us'],new.get('time_us')
 rows.append(dict(case=name,old_time_us=a,new_time_us=b,time_change_percent=(b/a-1)*100 if b is not None else None,text_identical=text,ds_identical=ds,status='compile_failed' if b is None else 'passed',details=details))
for bm in [32,64]:
 d=json.loads((r/f'gemm1-bm{bm}/comparison.json').read_text());pair={x['name']:x for x in d['rows']}
 add(f'GEMM1 BM{bm} BN{256 if bm==32 else 512}',pair['old'],pair['new'],f'gemm1-bm{bm}/comparison.json',d['identical_elf_text'],d['identical_static_ds_counts'])
d=json.loads((r/'gemm2-bm32/comparison.json').read_text());pair={x['compiler']:x for x in d}
add('GEMM2 BM32 BN512',pair['old'],pair['new'],'gemm2-bm32/comparison.json',pair['old']['text_sha256']==pair['new']['text_sha256'],pair['old']['static_ds_and_mls_instructions']==pair['new']['static_ds_and_mls_instructions'])
d=json.loads((r/'gemm2-bm64/comparison.json').read_text())
for bn in [256,512]:
 pair={x['compiler']:x for x in d if x['BN']==bn}
 add(f'GEMM2 BM64 BN{bn}',pair['old'],pair['new'],'gemm2-bm64/comparison.json',pair['old']['text_sha256']==pair['new']['text_sha256'],pair['old']['static_ds_and_mls_instructions']==pair['new']['static_ds_and_mls_instructions'])
for d in json.loads((r/'bm128-comparison.json').read_text()):
 add(d['case'].upper().replace('-',' '),d['old'],d['new'],'bm128-comparison.json',d['text_identical'],d['ds_identical'])
assert len(rows)==10,len(rows)
(r/'summary.json').write_text(json.dumps(dict(gc_ghz=1.8,cu_count=48,quantization='FP8 channel',comparison='matched frozen inputs and compilers',rows=rows),indent=2)+'\n')
lines=['\n## Matched old/new timings at 1.8 GHz\n','| Kernel | Old (us) | New (us) | Change | Native text / DS |','|---|---:|---:|---:|---|']
for d in rows:
 b=d['new_time_us'];new=f'{b:.6f}' if b is not None else 'compile failed';delta=f"{d['time_change_percent']:+.2f}%" if b is not None else '—';same='identical' if d['text_identical'] else ('not produced' if b is None else 'changed')
 lines.append(f"| {d['case']} | {d['old_time_us']:.6f} | {new} | {delta} | {same} |")
lines+=['','All nine successfully compiled new kernels passed their CPU-reference, padding','and untouched-output checks, and have zero final spills/private storage. BM128','GEMM1 BN128/BN256 also have byte-identical old/new outputs. BN512 failed during','compile before a GEMM dispatch. No performance value is assigned to that failure.','', 'GEMM1 BM128 BN128 loses 12 static `ds_read2st64_b64` instructions, gains24', '`ds_read_b64` and36 VALU. BN256 loses24 paired reads, gains48 single reads,', '108 VALU and2 SALU. These are static whole-kernel counts; they are not dynamic', 'iteration totals. Old/new input LLVM IR is identical for these two paths.','', 'A first-attempt BN512 compile reports role demands8/16/240/252, a4-aligned', 'sum516 exceeding the512 VGPR per-SIMD budget. The old executable succeeded',
 'after adjusting final quotas to16/24/236/236; these quotas are not the old',
 'actual live-register demands. The new Triton quota checker rejects', 'it before producing a launchable final kernel. The diagnostic assembly has', 'spills and is labelled **first attempt only**, not a usable code object.','', 'The other seven geometries preserve native `.text`, DS/MLS instructions and', 'matched-run time exactly. This does not establish that the missing address', 'recovery is unnecessary: BM128 GEMM1 demonstrates both instruction and', 'compilation regressions. Only the requested control-only patch remains applied.','']
p=r/'README.md';text=p.read_text().split('\n## Matched old/new timings at 1.8 GHz')[0];p.write_text(text+'\n'.join(lines))
