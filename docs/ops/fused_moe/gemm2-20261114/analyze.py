import json,re,sys,collections,shutil
from pathlib import Path
r=Path(sys.argv[1]);result=json.loads((r/'result.json').read_text());assert result['mismatches']==0
launch=json.loads((r/'launch.json').read_text());stats=(r/'m5out/0/0/stats.txt').read_text()
def val(k):
 m=re.search(r'^'+re.escape(k)+r'\s+(\S+)',stats,re.M);return float(m[1]) if m else None
origin=val('system.shaders.firstWaveStartTick');end=val('system.shaders.lastWaveEndTick')
rx=re.compile(r'\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)')
mm=collections.defaultdict(list);active={};physical={};waits=collections.defaultdict(list);saved={};start=[];finish=[];warm=0
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
 for line in f:
  if not any(s in line for s in ('ACTIVE:','v_mmac_f32_16x16x32_fp8_fp8','WAVE_START[',': s_endpgm',': s_abarrier_try_wait ',': r [S')):continue
  m=rx.match(line)
  if not m:continue
  full=int(m[2],16);w=full&65535;physical[w]=full>>16;op=m[4];t=(int(m[1])-origin)/period;key=(w,m[3])
  if op.startswith('WAVE_START['):start.append(t)
  elif op.startswith('s_endpgm'):finish.append(t)
  elif op.startswith('ACTIVE:'):active[w]=int(op.split()[-1],16)
  elif op.startswith('v_mmac_f32_16x16x32_fp8_fp8'):
   if active.get(w):mm[w].append(t)
   else:warm+=1
  elif op.startswith('s_abarrier_try_wait '):
   e=dict(op=op,reads={},time=t);waits[w].append(e);saved[key]=e
  elif key in saved and op.startswith('r [S'):
   q=re.match(r'r \[S(\d+)\]([0-9a-f]+)',op)
   if q:saved[key]['reads'].setdefault(int(q[1]),int(q[2],16))
def union(times):
 last=-1;busy=0
 for t in sorted(times):busy+=max(0,t+8-max(last,t));last=max(last,t+8)
 return busy
simds=[]
for simd in range(4):
 times=[t for w,ts in mm.items() if physical[w]==simd for t in ts];assert times
 busy=union(times);span=max(times)+8-min(times)
 simds.append(dict(simd=simd,instructions=len(times),busy_cycles=busy,first=min(times),last=max(times)+8,mmac_envelope_percent=100*busy/span,whole_dispatch_percent=100*busy/((end-origin)/period)))
audit={}
for w,events in waits.items():
 ids=[]
 for e in events:
  operand=e['op'].split(None,1)[1].split(',')[1].strip();ids.append(e['reads'][int(operand[1:])] if operand.startswith('s') else int(operand))
 audit[w]=dict(ids=ids,first_nonzero=ids[0]!=0,no_adjacent_repeat=all(a!=b for a,b in zip(ids,ids[1:])))
assert all(e['first_nonzero'] and e['no_adjacent_repeat'] for e in audit.values())
# Logical unrolled K-body: first A READY wait through last real MMAC.
# Include operand waits inside K; exclude epilogue intervals where neither
# consumer executes K. Combine overlapping C0/C1 windows per physical SIMD.
k_body = None
if launch["mode"] == "gluon":
    nk = launch["actual_K"] // launch["options"]["BLOCK_SIZE_K"]
    per_output = launch["options"]["BLOCK_SIZE_M"] * (launch["options"]["BLOCK_SIZE_N"] // 2) * launch["actual_K"] // (4 * 16 * 16 * 32)
    windows = collections.defaultdict(list)
    for w, times in mm.items():
        assert len(times) % per_output == 0
        nout = len(times) // per_output
        if launch.get("a_reuse_across_n", False):
            nn = launch["actual_N"] // launch["options"]["BLOCK_SIZE_N"]
            assert nout % nn == 0
            assert len(waits[w]) == nout // nn * (nn + 1) * nk
        else:
            assert len(waits[w]) == nout * 2 * nk
        for out in range(nout):
            if launch.get("a_reuse_across_n", False):
                m, n = divmod(out, nn)
                first_wait = m * (nn + 1) * nk + (0 if n == 0 else (n + 1) * nk)
            else:
                first_wait = out * 2 * nk
            windows[physical[w]].append((waits[w][first_wait]["time"], times[(out + 1) * per_output - 1] + 8))
    def merged_span(intervals):
        end = -1
        total = 0
        for a,b in sorted(intervals):
            total += max(0,b-max(a,end))
            end = max(end,b)
        return total
    denom = sum(merged_span(v) for v in windows.values())
    k_body = dict(percent=100 * sum(v["busy_cycles"] for v in simds) / denom, window_cycles_sum=denom,
                  definition="per SIMD union of logical K bodies: first operand-ready wait to last MMAC+8; row reuse includes A preload in the first N output; operand stalls included, epilogue-only gaps excluded")
res=dict(timing_scope_note="CU0 first-to-last-wave span and K-body only; complete 48-CU dispatch acceptance uses schema-2 model-metrics.json",k_body_mmac=k_body,time_us=(end-origin)/1e6,gc_period_ps=period,whole_dispatch_mmac_percent=sum(s['whole_dispatch_percent'] for s in simds)/4,mmac_envelope_percent=100*sum(s['busy_cycles'] for s in simds)/sum(s['last']-s['first'] for s in simds),simds=simds,active_mmac_counts={w:len(v) for w,v in mm.items()},exec0_mmac_count=warm,wait_id_audit=audit,lds_bank_conflict=val('system.shaders.CUs00.ldsBankConflict'),checked_elements=result['checked_elements'],scope=launch)
perfs=list((r/'m5out/0/0').glob('*_python*.perf'));assert len(perfs)==1
name=f"GEMM2_M4096_N2048_K256_BM128_BN{launch['options']['BLOCK_SIZE_N']}_BK{launch['options']['BLOCK_SIZE_K']}_GRID{launch['grid']}_OUTPUT{launch['measured_logical_tiles']}_{r.name}_CU0.perf"
shutil.copy2(perfs[0],r/name);res['perf']=str(r/name).replace('/zhenxin/','/home/zhenx/')
(r/'analysis.json').write_text(json.dumps(res,indent=2));print({k:res[k] for k in ('time_us','whole_dispatch_mmac_percent','mmac_envelope_percent','checked_elements','lds_bank_conflict','perf')})
