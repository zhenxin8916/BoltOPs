import importlib.util,runpy,sys
from pathlib import Path
source=Path(sys.argv.pop(1)).resolve()
name="boltops.fused_moe.gluon.gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128_256"
import boltops.fused_moe.gluon
spec=importlib.util.spec_from_file_location(name,source)
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
module.DEFAULT_LAUNCH_OPTIONS=dict(module.DEFAULT_LAUNCH_OPTIONS,BLOCK_SIZE_K=64)
# A local copy of the CPU-only probe exposes BK64 as an explicit test argument.
probe=Path('/zhenxin/BoltOPs/boltops/tools/gluon_moe_gemm2.py').read_text().replace("choices=(128, 256), default=256)","choices=(64, 128, 256), default=256)")
exec(compile(probe,'/zhenxin/BoltOPs/boltops/tools/gluon_moe_gemm2.py','exec'),dict(__name__='__main__',__file__='/zhenxin/BoltOPs/boltops/tools/gluon_moe_gemm2.py'))
