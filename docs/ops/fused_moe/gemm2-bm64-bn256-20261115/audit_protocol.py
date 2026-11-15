from pathlib import Path
import re,json,sys,collections
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)");es=collections.defaultdict(list);keys={}
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 for line in f:
  m=rx.match(line)
  if not m:continue
  t,w,i,o=int(m[1]),int(m[2],16)&65535,m[3],m[4]
  if o.startswith(('s_abarrier_try_wait ','s_abarrier_arrive ')):
   e=dict(t=t,op=o,reads={});es[w].append(e);keys[w,i]=e
  elif o.startswith('r [S') and (w,i) in keys:
   q=re.match(r'r \[S(\d+)\]([0-9a-f]+)',o)
   if q:keys[w,i]['reads'].setdefault(int(q[1]),int(q[2],16))
def imm(v,e):return e['reads'][int(v[1:])] if v.startswith('s') else int(v)
out={}
for w,events in es.items():
 if not 8<=w<16:continue
 wait=[];arr=[]
 for e in events:
  arg=[v.strip() for v in e['op'].split(None,1)[1].split(',')]
  if e['op'].startswith('s_abarrier_try_wait'):wait.append([imm(arg[1],e),imm(arg[2],e)])
  else:arr.append(imm(arg[1],e))
 # Initialization arrivals precede partition waits; inspect only each role's
 # actual consumption sequence and post-first-wait slot-credit returns.
 aid=[(i,p) for i,p in wait if i in (1,2)]
 for epoch,(i,p) in enumerate(aid):assert (i,p)==(1+(epoch&1),(epoch>>1)&1),(w,aid)
 mid=[(i,p) for i,p in wait if i in (12,0)]
 assert mid==[(12 if e%2==0 else 0,(e>>1)&1) for e in range(len(aid))],(w,mid,aid)
 bid=5 if w<12 else 7
 bs=[(i,p) for i,p in wait if i==bid]
 assert bs==[(bid,e&1) for e in range(len(bs))],(w,bs)
 assert len(bs)%len(aid)==0
 out[w]=dict(A_generations=aid,metadata_generations=mid,B_generations=bs,M_iterations=len(aid),N_per_M=len(bs)//len(aid))
assert len(out)==8
(r/'protocol-audit.json').write_text(json.dumps(out,indent=2));print(r.name,[(w,v['M_iterations'],v['N_per_M']) for w,v in out.items()])
