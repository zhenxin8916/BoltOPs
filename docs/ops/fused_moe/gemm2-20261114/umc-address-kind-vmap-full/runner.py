import importlib.util,runpy,sys
from pathlib import Path
source=Path(sys.argv.pop(1)).resolve()
name="boltops.fused_moe.gluon.gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128_256"
import boltops.fused_moe.gluon
spec=importlib.util.spec_from_file_location(name,source)
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
spec.loader.exec_module(module)
runpy.run_path(str(source.parent/"probe.py"),run_name="__main__")
