from perf_dispatch_span import read_dispatch_span
from pathlib import Path
import re,json,sys,collections,hashlib,struct,zlib
r=Path(sys.argv[1]);l=json.loads((r/'launch.json').read_text());l.update(BM=l['bm'],BN=l['bn'],BK=l['bk'],logical_output_tiles=l['total_tiles'],grid=[l['grid']]);perout=l['BM']*(l['BN']//2)*256//(16*16*32*4)
rx=re.compile(r'\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)');mm=collections.defaultdict(list);wave_starts=[];wave_ends=[];active={};phys={};prev={};waits=collections.defaultdict(list);warm=0
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',next(f))[1])
 for line in f:
  if not any(x in line for x in (': ACTIVE:',': v_mmac_',': s_',': buffer_',': ds_',': global_',': matrix_',': v_',': WAVE_START')):continue
  m=rx.match(line)
  if not m:continue
  w=int(m[2],16)&65535
  if (w & 255)>=16:continue
  t=int(m[1])/period;op=m[4];phys[w]=int(m[2],16)>>16
  if op.startswith('ACTIVE:'):active[w]=int(op.split()[-1],16);continue
  if op.startswith('WAVE_START'):wave_starts.append(int(m[1]));continue
  if op.startswith('s_endpgm'):wave_ends.append(int(m[1]))
  if re.match(r'(s_|v_|buffer_|ds_|global_|matrix_)',op):
   if w in prev:
    ot,oo,pc=prev[w]
    if oo.startswith(('s_waitcnt','s_abarrier_try_wait','s_ebarrier_sync')):waits[w].append(dict(cycles=t-ot,op=oo,pc=pc,tick_cycle=ot))
   prev[w]=(t,op,m[3])
  if op.startswith('v_mmac_f32_16x16x32_fp8_fp8') and 8 <= (w & 255) < 16:
   if active.get(w):mm[w].append(t)
   else:warm+=1
outs=min((len(mm[w])//perout for w in range(8,16)),default=0)
def union(xs):
 end=-1;busy=0
 for t in sorted(xs):busy+=max(0,t+8-max(end,t));end=max(end,t+8)
 return busy
rounds=[]
for o in range(outs):
 sims=[]
 for p in range(4):
  ws=[w for w in range(8,16) if phys[w]==p];assert len(ws)==2
  xs=[t for w in ws for t in mm[w][o*perout:(o+1)*perout]]
  sims.append(dict(simd=p,first=min(xs),last=max(xs)+8,busy=union(xs),span=max(xs)+8-min(xs)))
 rounds.append(sims)
gaps=[rounds[i+1][s]['first']-rounds[i][s]['last'] for i in range(outs-1) for s in range(4)]
res=dict(BM=l['BM'],BN=l['BN'],BK=l['BK'],completed_CU0_outputs=outs,MMAC_per_wave_per_output=perout,MMAC_counts={w:len(mm[w]) for w in range(8,16)},per_output_hot_percent=[100*sum(x['busy'] for x in row)/sum(x['span'] for x in row) for row in rounds],mean_tail_head_gap_cycles=sum(gaps)/len(gaps) if gaps else None,tail_head_gaps_cycles=gaps,rounds=rounds,top_wait_issue_to_next_instruction_cycles={w:sorted(es,key=lambda x:-x['cycles'])[:10] for w,es in waits.items()},EXEC0_warmup=warm,partial=not (r/'result.json').exists())
if not res['partial']:
 n=json.loads((r/'result.json').read_text());assert n['verified'] and n['mismatches']==0 and n['untouched_tail']
 assert n['checked_elements']>0
 stats=(r/'m5out/0/0/stats.txt').read_text();v=lambda k:float(re.search(r'^'+re.escape(k)+r'\s+(\S+)',stats,re.M)[1]);start=min(wave_starts);end=max(wave_ends)+period;span=(end-start)/period
 assert all(len(mm[w])==((l['logical_output_tiles']+l['grid'][0]-1)//l['grid'][0])*perout for w in range(8,16))
 res.update(time_us=read_dispatch_span(r)['duration_us'],wave_span_us=(end-start)/1e6,CU0_overall_MMAC_percent=100*sum(union([t for w in mm if phys[w]==p for t in mm[w]]) for p in range(4))/(4*span),numeric=n)
 counts={int(m[0]):int(m[1]) for m in re.findall(r'^system.shaders.CUs(\d+)\.numInstrTypeExecuted::MMOP\s+(\d+)',stats,re.M)}
 expected=96*l['BM']*l['BN']*256*2//16384
 assert len(counts)==48 and sum(counts.values())==expected
 assert sum(len(v) for v in mm.values())==counts[0]
 assert warm==0
 res.update(whole_GPU_MMAC_percent=100*sum(counts.values())*8/(48*4*(res['time_us']*1e6/period)),expected_all_CU_MMAC=expected,all_CU_MMAC_counts=counts,weighted_hot_percent=100*sum(x['busy'] for row in rounds for x in row)/sum(x['span'] for row in rounds for x in row))
 perfs=list(r.glob('m5out/0/0/*_python*.perf'));assert len(perfs)==1
 target=r/f'GEMM2_M128_N2048_K256_BM{l["BM"]}_BN{l["BN"]}_BK{l["BK"]}_GRID48_OUTPUT96_2TILEperTG_GC{round(1e6/period/100)*100}_CU0.perf';target.write_bytes(perfs[0].read_bytes());res['perf']=str(target)
 tt=list(r.glob('m5out/0/0/*ttrace_model__xcd0_cu0_se0.out'));assert len(tt)==1
 raw=b''.join(struct.pack('<H',int(x,16)) for x in tt[0].read_text().split());pb=target.read_bytes();match=False
 for i in range(len(pb)-1):
  if pb[i]!=120 or pb[i+1] not in (1,156,218):continue
  try:d=zlib.decompress(pb[i:])
  except zlib.error:continue
  if d==raw:match=True;break
 assert match;res['perf_Wavefront_TT_verified']=True
(r/('partial-analysis.json' if res['partial'] else 'analysis.json')).write_text(json.dumps(res,indent=2))
print(r.name,{k:res[k] for k in ('completed_CU0_outputs','per_output_hot_percent','mean_tail_head_gap_cycles','partial','MMAC_counts')},res.get('time_us'),flush=True)
