import re,json,sys,collections
from pathlib import Path
r=Path(sys.argv[1]);events=collections.defaultdict(list);current={}
rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)")
for line in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 if 's_abarrier_try_wait ' not in line and ': r [S' not in line:continue
 m=rx.match(line)
 if not m:continue
 w=int(m[2],16);key=(w,m[3]);op=m[4]
 if op.startswith('s_abarrier_try_wait '):
  e={'op':op,'reads':{}};events[w].append(e);current[key]=e
 elif key in current:
  q=re.match(r'r \[S(\d+)\]([0-9a-f]+)',op)
  if q:current[key]['reads'].setdefault(int(q[1]),int(q[2],16))
def value(s,e):
 s=s.strip();return e['reads'][int(s[1:])] if s.startswith('s') else int(s,0)
out={}
for w,es in events.items():
 if (w&65535)<8:continue
 tokens=[];bs=[];aa=[];scales=[]
 for e in es:
  args=e['op'].split(None,1)[1].split(',');bid=value(args[1],e);ph=value(args[2],e)
  if bid==2:scales.append([bid,ph])
  if bid in (5,7):bs.append([bid,ph])
  if bid==1:aa.append(ph)
 assert tokens==[],(w,tokens)
 assert aa==[0,1],(w,aa)
 expected=([[5,0],[5,1]] * 2) if (w&65535)<12 else ([[7,0],[7,1]] * 2)
 assert bs==expected,(w,bs)
 expected_scales=[[2,0],[2,1]]
 assert scales==expected_scales,(w,scales)
 out[f'{w:08x}']={'token_phases':tokens,'a_phases':aa,'b_id_phase':bs,'scale_ready_id_phase':scales}
assert len(out)==8
(r/'tile-sb-phase-audit.json').write_text(json.dumps(out,indent=2)+'\n');print('8 consumer waves: A/B and separate scale READY generations verified')
