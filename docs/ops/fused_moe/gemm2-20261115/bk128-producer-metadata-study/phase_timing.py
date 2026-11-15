import sys,re,json,collections
from pathlib import Path
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)");mm=collections.defaultdict(list);ready=collections.defaultdict(list);stores=collections.defaultdict(list);period=None
for line in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 if period is None:period=int(re.search(r'GC_CLOCK (\d+)',line)[1]);continue
 if not any(x in line for x in [': v_mmac_',': s_abarrier_arrive ',': buffer_store_']):continue
 m=rx.match(line)
 if not m:continue
 t=int(m[1]);full=int(m[2],16);w=full&65535;op=m[3]
 if op.startswith('v_mmac_'):mm[full].append(t)
 if op.startswith('s_abarrier_arrive') and w>=8 and (', 9,' in op or ', 13,' in op):ready[full].append(t)
 if op.startswith('buffer_store_'):stores[full].append(t)
out={'period_ps':period,'definition':'CU0 cycles, per consumer 128 MMAC instructions per independent M output. Own inter-M gap includes old postprocessing and next data preparation; postprocess-to-READY includes DS completion and scheduling, not pure VALU cycles.','consumers':[],'stores':[]}
for full,ts in sorted(mm.items()):
 assert len(ts)==256,(full,len(ts));rs=ready[full];assert len(rs)%2==0;nr=len(rs)//2
 out['consumers'].append(dict(simd=full>>16,role='C0' if (full&65535)<12 else 'C1',m0_first_mmac_tick=ts[0],m0_last_mmac_end_tick=ts[127]+8*period,m1_first_mmac_tick=ts[128],inter_m_gap_cycles=(ts[128]-ts[127])/period-8,m0_post_mmac_to_final_ready_cycles=(rs[nr-1]-ts[127])/period-8,m1_post_mmac_to_final_ready_cycles=(rs[-1]-ts[-1])/period-8))
for full,ts in sorted(stores.items()):
 assert len(ts) in (32,64),(full,len(ts));ns=len(ts)//2;out['stores'].append(dict(simd=full>>16,store_instructions_per_M=ns,m0_first_store_tick=ts[0],m0_last_store_tick=ts[ns-1],m1_first_store_tick=ts[ns],m1_last_store_tick=ts[-1],m0_last_to_m1_first_store_cycles=(ts[ns]-ts[ns-1])/period))
(r/'phase-timing.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
