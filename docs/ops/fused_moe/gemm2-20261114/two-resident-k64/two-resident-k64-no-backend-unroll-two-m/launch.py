import os,subprocess,sys
from pathlib import Path
r=Path(__file__).parent
triton='/zhenxin/temp/moe-gemm2-20261114/mls-stable-groups/triton'
env=dict(os.environ,PYTHONPATH=triton+'/python:/zhenxin/BoltOPs',TRITON_BACKENDS_IN_TREE='1',TRITON_ALWAYS_COMPILE='1',TRITON_CACHE_DIR=str(r/'cache'),GPU_CHIP='sb',GPU_DFLAGS="['StatLog','IT0','TT0','Perf','UMC']",GPU_ARGS="['--gpu-clock=2.2GHz']",PMD_PATH='/opt/rocm/pmd',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',TRITON_HIP_CLANG_PATH='/zhenxin/temp/moe-gemm2-20261114/clang-control-sync-no-unroll')
cmd=['setarch','x86_64','-R','/zhenxin/temp/hcu-gluon-upstream-venv-20261103/bin/python',str(r/'run_k64.py'),str(r/'gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128_256.py'),'--bn','256','--bk','128','--tiles','384']+sys.argv[1:]
with (r/'run.log').open('w') as f:
 p=subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
(r/'exit-code.txt').write_text(str(p.returncode));print((r/'run.log').read_text()[-6500:]);sys.exit(p.returncode)
