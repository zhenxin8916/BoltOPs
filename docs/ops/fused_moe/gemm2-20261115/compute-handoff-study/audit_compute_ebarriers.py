import re,json,sys,collections
from pathlib import Path
r=Path(sys.argv[1]);groups=int(sys.argv[2]);events=collections.defaultdict(list);cur={};simds={}
rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)")
for line in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 if 's_ebarrier_' not in line and ': r [S' not in line:continue
 m=rx.match(line)
 if not m:continue
 w=int(m[2],16);key=(w,m[3]);op=m[4];simds[w]=w>>16
 if op.startswith(('s_ebarrier_sync ','s_ebarrier_arrive ')):
  e={'op':op,'reads':{}};events[w].append(e);cur[key]=e
 elif key in cur:
  q=re.match(r'r \[S(\d+)\]([0-9a-f]+)',op)
  if q:cur[key]['reads'].setdefault(int(q[1]),int(q[2],16))
def val(s,e):
 s=s.strip();return e['reads'][int(s[1:])] if s.startswith('s') else int(s,0)
out={};participants=collections.defaultdict(set)
for w,es in events.items():
 role=(w&65535)//4
 expected_sync=(14+role-2) if groups==1 else 10+2*((w&3)//2)+role-2
 expected_arrive=(15-(role-2)) if groups==1 else 11+2*((w&3)//2)-(role-2)
 manual=[]
 for e in es:
  a=e['op'].split(None,1)[1].split(',');bid=val(a[1],e);count=val(a[2],e);assert 0<=bid<16
  if bid>=10:
   assert role in (2,3),(w,e)
   sync='sync' in e['op'];assert bid==(expected_sync if sync else expected_arrive),(w,bid,expected_sync,expected_arrive)
   assert count==(8 if groups==1 else 4),(w,count)
   manual.append([bid,count,'sync' if sync else 'arrive']);participants[bid].add(w)
 if role in (2,3):
  assert sum(x[2]=='sync' for x in manual)==12,(w,manual)
  assert sum(x[2]=='arrive' for x in manual)==(12 if role==2 else 13),(w,manual)
  out[f'{w:08x}']={'simd':simds[w],'manual_events':manual}
assert len(out)==8
assert len(participants)==2*groups
assert all(len(ws)==(8//groups) for ws in participants.values())
for ws in participants.values():
 assert len({simds[w] for w in ws})==4//groups
 assert sum((w&65535)//4==2 for w in ws)==4//groups
 assert sum((w&65535)//4==3 for w in ws)==4//groups
(r/'compute-ebarrier-audit.json').write_text(json.dumps({'groups':groups,'range_0_15':True,'compiler_ids_reserved':'0..9','manual_ids':sorted(participants),'participants':{str(k):[f'{w:08x}' for w in sorted(v)] for k,v in participants.items()},'waves':out},indent=2)+'\n');print('EB IDs, wave groups, rendezvous count and two-M handoff counts verified')
