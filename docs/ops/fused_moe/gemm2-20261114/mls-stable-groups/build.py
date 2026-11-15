import subprocess
from pathlib import Path
r=Path(__file__).parent
b='/zhenxin/worktrees/triton-feature-20261102/build/cmake.linux-x86_64-cpython-3.10'
with (r/'build.log').open('w') as f:
 for name in ['compile-command.txt','link-command.txt']:
  subprocess.run((r/name).read_text(),shell=True,cwd=b,stdout=f,stderr=subprocess.STDOUT,check=True)
print('stable MLS isolated build complete')
