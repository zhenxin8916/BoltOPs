import sys,re,json,collections
from pathlib import Path
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: ((?:s_|v_|ds_|buffer_|matrix_).*)$")
by=collections.defaultdict(list)
for l in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 m=rx.match(l)
 if m:by[int(m[2],16)].append((int(m[1]),m[3]))
period=556;out={'definition':'IT issue-to-next-issue gaps include scheduling and dependencies, not isolated hardware instruction latency. GM credit counters are whole-CU across loads, MLS and stores, not store-only.','period_ps':period,'p0':[]};allstores=[]
for w,es in sorted(by.items()):
 if not any(o.startswith('buffer_store_') for t,o in es):continue
 st=[i for i,(t,o) in enumerate(es) if o.startswith('buffer_store_')];allstores += [es[i][0] for i in st]
 ready=[dict(op=o,gap_cycles=(es[i+1][0]-t)/period) for i,(t,o) in enumerate(es[:-1]) if 'abarrier_try_wait' in o and (', 9,' in o or ', 13,' in o)]
 groups=[]
 for j in range(0,len(st),4):
  ix=st[j:j+4];a=max(i for i in range(ix[0]) if 'abarrier_try_wait' in es[i][1]);h=collections.Counter(o.split()[0] for t,o in es[a:ix[-1]+1]);groups.append(dict(index=j//4,first_to_last_store_cycles=(es[ix[-1]][0]-es[ix[0]][0])/period,ready_to_last_store_cycles=(es[ix[-1]][0]-es[a][0])/period,salu_instructions=sum(v for k,v in h.items() if k.startswith('s_')),valu_instructions=sum(v for k,v in h.items() if k.startswith('v_')),ds_instructions=sum(v for k,v in h.items() if k.startswith('ds_'))))
 out['p0'].append(dict(full_wave=w,simd=w>>16,store_instructions=len(st),ready_waits=ready,chunks=groups))
allstores.sort();out['all_simd_store_interval_histogram_cycles']={str(k):v for k,v in collections.Counter((b-a)/period for a,b in zip(allstores,allstores[1:])).items()}
out['whole_cu_metrics']=json.loads((r/'model-metrics.json').read_text())['cu_metrics']
(r/'store-issue-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['p0'][0],indent=2))
