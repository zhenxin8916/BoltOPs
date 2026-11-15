import re,json,sys,collections
from pathlib import Path
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)");hist=collections.defaultdict(collections.Counter)
for line in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 if not any(x in line for x in [': buffer_',': global_',': v_permlane',': ds_read_']):continue
 m=rx.match(line)
 if m:hist[int(m[2],16)&65535][m[3].split()[0]]+=1
out={}
for w,c in sorted(hist.items()):
 role='P0' if w<4 else 'P1' if w<8 else 'C0' if w<12 else 'C1'
 if w>=8:assert sum(v for k,v in c.items() if k.startswith(('buffer_load','global_load','v_permlane')))==0,(w,c)
 out[w]=dict(role=role,instruction_counts=dict(c))
assert sum(w>=8 for w in out)==8
(r/'consumer-no-vmem-audit.json').write_text(json.dumps(out,indent=2)+'\n');print('All eight CU0 consumer waves have zero buffer/global loads and zero permlane instructions.')
