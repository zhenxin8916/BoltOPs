import re,json,sys,collections
from pathlib import Path
r=Path(sys.argv[1]);by=collections.defaultdict(list)
rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)")
for line in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 m=rx.match(line)
 if m and (int(m[2],16)&65535)>=8 and re.match(r'(s_|ds_)',m[3]):by[int(m[2],16)].append((int(m[1]),m[3]))
out={}
for w,events in sorted(by.items()):
 groups=[];start=None
 for i,(t,op) in enumerate(events):
  if re.match(r's_abarrier_try_wait \w+, 1, ',op):start=i
  if re.match(r's_abarrier_arrive \w+, 3, ',op):
   assert start is not None,(w,i)
   seq=events[start:i];ds=[(t,o) for t,o in seq if o.startswith('ds_read')];waits=[(t,o) for t,o in seq if o.startswith('s_waitcnt') and 'lgkmcnt(0)' in o]
   assert sum('ds_read_matrix_trans_format' in o for t,o in ds)==24,(w,ds)
   assert sum(o.startswith('ds_read2_b64 ') for t,o in ds)==1,(w,ds)
   assert len(waits)==1 and waits[0][0]>ds[-1][0],(w,waits,ds[-1])
   expected=7 if (w&65535)>=12 else 5
   bread=[t for t,o in seq if re.match(r's_abarrier_try_wait \w+, '+str(expected)+r', ',o)]
   assert len(bread)==1 and ds[15][0]<bread[0]<ds[16][0],(w,bread)
   groups.append(dict(matrix_reads=24,sb_read2_b64=1,wait0_count=1,all_reads_before_wait0=True,A_EMPTY_after_wait0=True))
   start=None
 assert len(groups)==2,(w,len(groups));out[f'{w:08x}']=groups
assert len(out)==8,len(out)
(r/'combined-ds-wait-audit.json').write_text(json.dumps(out,indent=2)+'\n');print('8 consumer waves, two M each: A16 + B0 8 matrix DS reads + SB read before one wait0 and A EMPTY')
