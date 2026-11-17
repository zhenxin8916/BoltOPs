from pathlib import Path
import re,json,collections,sys
r=Path(sys.argv[1]);l=json.loads((r/'launch.json').read_text());src=(r/'kernel_snapshot.py').read_text();panels=4;panels*=l['N_TILES_LIMIT'] or l['options']['N']//l['bn'];b2='[2, K, BN // 8]' in src
rx=re.compile(r'\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)');ev={};regs={};counts=collections.defaultdict(collections.Counter);lastread={};lastwrite={};lastwait={};bad=[];seq={};loads=collections.defaultdict(list);pendingloads=collections.Counter()
def number(s,rr):
 s=s.strip();return rr[int(s[1:])] if s.startswith('s') else int(s)
first_wait={};last_barrier_wait={};wait_total=collections.Counter()
def commit(w,op,t,rr):
 if op.startswith('s_abarrier_try_wait '):
  v=number(op.split(None,1)[1].split(',')[1],rr)
  assert 0<=v<=15,(w,v)
  if w not in first_wait:first_wait[w]=v;assert v!=0,(w,v)
  assert last_barrier_wait.get(w)!=v,(w,v,'same consecutive barrier wait')
  last_barrier_wait[w]=v;wait_total[w]+=1

 if op.startswith('ds_read'):lastread[w]=t
 if op.startswith('ds_write'):lastwrite[w]=t
 if op.startswith('s_waitcnt ') and 'lgkmcnt(0)' in op:lastwait[w]=t
 if op.startswith('s_abarrier_seq '):
  args=op.split(None,1)[1].split(',');v=number(args[-1],rr);assert w not in seq,(w,seq[w],v);seq[w]=v;pendingloads[w]=0
 if op.startswith('matrix_load_'):
  assert w in seq,(w,op);pendingloads[w]+=1
 if not op.startswith('s_abarrier_arrive '):return
 args=op.split(None,1)[1].split(',');v=number(args[1],rr);counts[w][v]+=1
 if w in seq and v==seq[w]:
  assert pendingloads[w]>0,(w,v);loads[w].append(dict(id=v,native_MLS=pendingloads[w]));del seq[w]
 if w>=8 and v in (3,6,8,14,4) or w<4 and v in (11,15):
  if lastwait.get(w,-1)<lastread.get(w,-1):bad.append((w,op,'read not completed'))
 if w>=8 and v in (9,13):
  if lastwait.get(w,-1)<lastwrite.get(w,-1):bad.append((w,op,'write not completed'))
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 for line in f:
  m=rx.match(line)
  if not m:continue
  w=int(m[2],16)&65535
  if w>=16:continue
  op=m[4];t=int(m[1])
  if re.match(r'(s_|v_|ds_|matrix_|buffer_|global_)',op):
   if w in ev:commit(w,*ev[w],regs[w])
   ev[w]=(op,t);regs[w]={}
  elif op.startswith('r [S'):
   q=re.match(r'r \[S(\d+)\]([0-9a-f]+)',op)
   if q:regs[w].setdefault(int(q[1]),int(q[2],16))
for w in ev:commit(w,*ev[w],regs[w])
assert not bad,bad;assert not seq,seq
for w in range(8,16):
 assert counts[w][3]==2,(w,counts[w])
 be=(6,14) if w<12 and b2 else (8,4) if b2 else (6,) if w<12 else (8,)
 assert sum(counts[w][i] for i in be)==2*panels,(w,counts[w])
 assert counts[w][9 if w<12 else 13]==2*panels,(w,counts[w])
for w in range(4):
 assert counts[w][1]==2 and counts[w][12]==1 and counts[w][0]==1,(w,counts[w])
 assert counts[w][11]==counts[w][15]==1+2*panels,(w,counts[w])
 assert len(loads[w])==2,(w,loads[w])
for w in range(4,8):assert len(loads[w])==4*panels,(w,loads[w])
out=dict(first_wait_by_wave=first_wait,total_waits_by_wave=dict(wait_total),all_first_wait_ids_nonzero=True,consecutive_wait_ids_distinct=True,maximum_barrier_id=15,A_generations_per_reader=2,B_and_output_generations_per_reader=2*panels,B_double_buffered=b2,root_output_EMPTY_seed_per_P0_wave=1,full_LDS_completion_before_EMPTY_and_output_READY=True,all_native_MLS_tracked_by_seq_and_arrive=True,counts={w:dict(v) for w,v in counts.items()},producer_MLS_tracks=loads)
(r/'pipeline-audit.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k not in ('counts','producer_MLS_tracks')})
