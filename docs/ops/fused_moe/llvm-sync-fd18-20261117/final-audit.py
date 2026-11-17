from pathlib import Path
import hashlib,json,subprocess
r=Path('/zhenxin/BoltOPs/docs/ops/fused_moe/llvm-sync-fd18-20261117')
s=json.loads((r/'summary.json').read_text());rows=s['rows']
assert len(rows)==10
assert sum(x['text_identical'] is True for x in rows)==7
assert sum(x['status']=='compile_failed' for x in rows)==1
assert sum(x['time_change_percent'] is not None and x['time_change_percent']>0 for x in rows)==2
b=json.loads((r/'bm128-comparison.json').read_text())
for x in b:
 for version in ['old','new']:
  d=x[version]
  if d.get('compile_failed'):continue
  assert d['result']['verified'] and d['result']['mismatches']==0 and d['result']['untouched_tail']
  assert len(d['spill_fields'])==3 and set(d['spill_fields'])=={'0'},d
 if not x['new'].get('compile_failed'):
  assert x['output_identical']
  if x['case'].startswith('gemm1-'):assert x['llvm_ir_identical']
llvm=Path('/zhenxin/llvm-project')
def git(*a):return subprocess.check_output(['git','-C',str(llvm),*a],text=True).strip()
assert git('status','--porcelain')==''
assert git('rev-list','--count','origin/dtk-llvm-6.3.3..HEAD')=='1'
base=Path('/zhenxin/temp/llvm-sync-fd18-20261117')
assert hashlib.sha256((base/'new-bin/clang-18').read_bytes()).digest()==hashlib.sha256((llvm/'build-dtk-llvm-6.3.3/bin/clang-18').read_bytes()).digest()
p=json.loads((r/'provenance.json').read_text());p['new_frozen_compiler_path']=str(base/'new-bin/clang-18');p['new_frozen_linker_path']=str(base/'new-bin/ld.lld');p['old_frozen_compiler_path']=str(base/'old-bin/clang-18');p['triton_commit']=subprocess.check_output(['git','-C','/zhenxin/worktrees/triton-feature-20261102','rev-parse','HEAD'],text=True).strip();p['triton_tracked_changes']=subprocess.check_output(['git','-C','/zhenxin/worktrees/triton-feature-20261102','diff','HEAD','--stat'],text=True).strip();assert not p['triton_tracked_changes'];(r/'provenance.json').write_text(json.dumps(p,indent=2)+'\n')
(r/'final-audit.json').write_text(json.dumps(dict(configurations=10,successful_new_gpu_runs=9,identical_native_code_and_time=7,gemm1_bm128_time_regressions=2,new_compile_rejections=1,all_successful_final_hsacos_spill_free=True,llvm_worktree_clean=True,only_requested_patch_above_upstream=True,frozen_new_compiler_matches_active=True,triton_tracked_changes=False),indent=2)+'\n')
manifest={str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(r.rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'}
(r/'artifact-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n');print('FINAL_AUDIT_PASS',len(manifest),'artifacts')
