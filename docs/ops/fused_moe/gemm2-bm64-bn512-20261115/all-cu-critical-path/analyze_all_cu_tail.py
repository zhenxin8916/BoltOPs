import re,json,sys,collections
from pathlib import Path
root=Path(sys.argv[1]);out=[]
rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)")
start=re.compile(r"WAVE_START\[CU(\d+)_SIMD(\d+)_WAVE(\d+)\]")
for trace in sorted((root/'m5out/0/0').glob('itrace_cu*_se*_xcd0.txt')):
 waves={};roles=collections.defaultdict(lambda:collections.defaultdict(list));cu=None
 with trace.open() as f:
  for line in f:
   m=rx.match(line)
   if not m: continue
   tick=int(m[1]);w=int(m[2],16);op=m[3];v=start.search(op)
   if v:
    cu=int(v[1]);waves[w]=(int(v[2]),int(v[3]));assert int(v[3])<4
    roles[int(v[3])]['start'].append(tick);continue
   if w not in waves:continue
   role=waves[w][1]
   if op.startswith('v_mmac_'):roles[role]['mmac'].append(tick)
   if op.startswith('buffer_store_'):roles[role]['store'].append(tick)
   if op.startswith('s_endpgm'):roles[role]['end'].append(tick)
   if op.startswith('s_abarrier_try_wait'):roles[role]['ab_wait'].append((tick,op))
   if op.startswith('s_ebarrier_sync'):roles[role]['eb_sync'].append((tick,op))
 assert cu is not None and len(waves)==16,(trace,len(waves))
 d={'cu':cu,'trace':str(trace),'roles':{}}
 for k,v in roles.items():
  assert len(v['start'])==len(v['end'])==4,(trace,k)
  d['roles'][k]={}
  for family in ['start','end','mmac','store']:
   a=v.get(family,[])
   if a:d['roles'][k][family]={'first_tick':min(a),'last_tick':max(a),'count':len(a)}
 d['last_store_tick']=max(v.get('store',{}).get('last_tick',0) for v in d['roles'].values())
 d['last_endpgm_tick']=max(v['end']['last_tick'] for v in d['roles'].values())
 d['last_mmac_tick']=max(v.get('mmac',{}).get('last_tick',0) for v in d['roles'].values())
 out.append(d)
assert len(out)==48 and len({v['cu'] for v in out})==48
out.sort(key=lambda x:x['last_endpgm_tick'],reverse=True)
(root/'all-cu-tail.json').write_text(json.dumps(out,indent=2)+'\n')
print([(x['cu'],x['last_mmac_tick']/1e6,x['last_store_tick']/1e6,x['last_endpgm_tick']/1e6) for x in out[:10]])
