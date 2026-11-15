from pathlib import Path
import json,hashlib,struct,sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from analyze import analyze
d=root/'perf-fixed/new-bottom-bn256-fp8-gc2200'
canonical=root/'perf/new-bottom-bn256-fp8-gc1800'
m=analyze(d)
a=json.loads((d/'cache-before.json').read_text());b=json.loads((d/'cache-after.json').read_text())
assert a==b,'Cache not untouched'
assert (canonical/'gemm/kernel.hsaco').read_bytes()==(d/'gemm/kernel.hsaco').read_bytes()
la=json.loads((canonical/'launch.json').read_text());lb=json.loads((d/'launch.json').read_text())
ra=json.loads((canonical/'result.json').read_text());rb=json.loads((d/'result.json').read_text())
assert la['input_sha256']==lb['input_sha256']
assert ra['output_sha256']==rb['output_sha256']
def text_sha(p):
 data=p.read_bytes();off=struct.unpack_from('<Q',data,40)[0];size,num,st=struct.unpack_from('<HHH',data,58);hdr=[struct.unpack_from('<IIQQQQIIQQ',data,off+i*size) for i in range(num)];h=hdr[st];strings=data[h[4]:h[4]+h[5]]
 for h in hdr:
  if strings[h[0]:].split(b'\0')[0]==b'.text':return hashlib.sha256(data[h[4]:h[4]+h[5]]).hexdigest()
report={'cache_hit':True,'cache_file_count':len(a),'cache_hashes_and_mtimes_unchanged':a==b,'canonical_hsaco_exact_match':True,'canonical_text_sha256':text_sha(canonical/'gemm/kernel.hsaco'),'retest_text_sha256':text_sha(d/'gemm/kernel.hsaco'),'input_hashes_equal':True,'output_hash_equal':True,'result':rb,'metrics':m}
(d/'canonical-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:m[k] for k in ('time_us','mmac_total_pct','hot_loop_mmac_pct')}))
