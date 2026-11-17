import collections, hashlib, json, re, shutil, subprocess
from pathlib import Path
r=Path('/zhenxin/temp/llvm-sync-fd18-20261117');doc=Path('/zhenxin/BoltOPs/docs/ops/fused_moe/llvm-sync-fd18-20261117')
def git(*args):return subprocess.check_output(['git','-C','/zhenxin/llvm-project',*args],text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def funcs(s):
 out={}
 for m in re.finditer(r'^([A-Za-z_][\w]*):.*?^\s*\.Lfunc_end\d+:',s,re.M|re.S):out[m[1]]=dict(collections.Counter(re.findall(r'^\s+(ds_\w+)',m[0],re.M)))
 return out
rows=[]
for test in ['lds-common-base','lds-wrapped-constant-read2']:
 old=funcs((r/'probes'/f'{test}.old.s').read_text());new=funcs((r/'probes'/f'{test}.new.s').read_text())
 rows.append(dict(test=test,changed_functions=[dict(name=n,old=old[n],new=new[n]) for n in old if old[n]!=new[n]]))
(doc/'ds-pairing-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
provenance=dict(old_source_checkout='89bee503d8e9d0bdcfbb4d97e204d5e346f02550',old_compiler_build_revision='fd18eddbaf4a662ae5ce86c516d28e5b16952513',old_clang_sha256=sha(r/'old-bin/clang-18'),new_source_head=git('rev-parse','HEAD'),upstream_head=git('rev-parse','origin/dtk-llvm-6.3.3'),tracking_branch='dtk-llvm-6.3.3',retained_patch='fd18eddbaf4a662ae5ce86c516d28e5b16952513',new_clang_sha256=sha(Path('/zhenxin/llvm-project/build-dtk-llvm-6.3.3/bin/clang-18')),backup_branch='backup/dtk-llvm-before-sync-20261117-89bee503',external_backup_path=str(r),worktree_status=git('status','--porcelain=v1'),local_commits=git('log','--format=%H %s','origin/dtk-llvm-6.3.3..HEAD'),build_target='clang llc opt FileCheck llvm-objdump llvm-readobj lld',control_only_flag_enabled_for_both=True,matrix_store_protection_patch_retained=False,ds_address_recovery_patch_retained=False,notes='Old source checkout and executable build identity differ; frozen executable hashes define the old numerical/performance baseline. No comparison claims that old clang included source-only 89bee503 matrix-store protection.')
(doc/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
for n in ['validation.json','validate.py']:
 shutil.copy2(r/n,doc/n)
