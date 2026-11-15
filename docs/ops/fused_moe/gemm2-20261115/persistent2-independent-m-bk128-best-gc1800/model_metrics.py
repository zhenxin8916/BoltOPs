import json,re,sys,statistics
from pathlib import Path
from perf_dispatch_span import read_dispatch_span
import configparser
r=Path(sys.argv[1]);result=json.loads((r/'result.json').read_text());assert result['mismatches']==0
launch=json.loads((r/'launch.json').read_text());s=(r/'m5out/0/0/stats.txt').read_text()
def val(key):
 m=re.search(r'^'+re.escape(key)+r'\s+(\S+)',s,re.M)
 return float(m[1]) if m else None
config=configparser.ConfigParser();config.read(r/'m5out/config.ini')
cu_config=[sec for sec in config.sections() if re.fullmatch(r'system\.shaders\.CUs\d+',sec)]
assert len(cu_config)==48
assert all(int(config[sec]['num_simds'])==4 and int(config[sec]['wf_size'])==64 for sec in cu_config)
period=int(config[config['system.shaders']['clk_domain']]['clock'])
trace=r/'m5out/0/0/itrace_cu0_se0_xcd0.txt'
if trace.exists():
 assert int(re.search(r'GC_CLOCK (\d+)',trace.open().readline())[1])==period
cycles=(val('system.shaders.lastWaveEndTick')-val('system.shaders.firstWaveStartTick'))/period
families=set(re.findall(r'\b(v_mmac_\w+)',(r/'gemm/kernel.amdgcn').read_text()))
assert families=={'v_mmac_f32_16x16x32_fp8_fp8'},families
counts={int(m[0]):int(m[1]) for m in re.findall(r'^system.shaders.CUs(\d+)\.numInstrTypeExecuted::MMOP\s+(\d+)',s,re.M)}
assert len(counts)==48,len(counts)
assert launch['mode'] in ('gluon','triton')
n=launch['measured_logical_tiles'];bm=launch['options']['BLOCK_SIZE_M'];bn=launch['options']['BLOCK_SIZE_N'];k=launch['actual_K']
expected=n*bm*bn*k*2//16384
assert sum(counts.values())==expected,(sum(counts.values()),expected)
if (r/'analysis.json').exists():
 a=json.loads((r/'analysis.json').read_text());assert a['exec0_mmac_count']==0
 assert abs(sum(v['busy_cycles'] for v in a['simds'])-counts[0]*8)<1e-6
cache={key:val('system.shaders.cache_system.'+key) for key in ['TccSendRespBandwidth','TccRecvRespBandwidth','TCCHitRate','TCPCacheLineHitRate','SQCIHitRate','missSQCIRequests','missDuplicatedSQCIRequests']}
metrics={}
for key in ['ldsBankConflict','GlobalMemPipeline.cmdCreditStall','GlobalMemPipeline.addrCreditStall','GlobalMemPipeline.dataCreditStall','LocalMemPipeline.creditStall','tatd.TATD.taSendStallBytcp','tatd.TATD.taSendStallByCredit','numCyclesWaitIFetchAfterBranchTaken']:
 values=[val(f'system.shaders.CUs{cu:02d}.'+key) for cu in range(48)]
 if all(v is not None for v in values):metrics[key]=dict(mean=statistics.mean(values),maximum=max(values),sum=sum(values))
wave_span_time_us=cycles*period/1e6
wave_span_mmac_percent=100*sum(counts.values())*8/(48*4*cycles)
span=read_dispatch_span(r)
assert val('system.shaders.dispatcher.numKernelLaunched')==1
assert abs(span['duration_ticks']-val('totalSimTicks'))<=max(1000,period)
time_us=span['duration_us']
cycles=span['duration_ticks']/period
out=dict(schema_version=2,time_us=time_us,wave_span_time_us=wave_span_time_us,wave_span_mmac_percent=wave_span_mmac_percent,dispatch_span=span,whole_gpu_mmac_percent=100*sum(counts.values())*8/(48*4*cycles),definition='all 48 CU MMOP counts * 8 cycles / (48 CU * 4 SIMD * full HYGPERF XCD time-section cycles, including startup); FP8 16x16x32 geometry/work audited. Schema 1 incorrectly called the narrower first-to-last-wave interval whole dispatch; legacy values retained as wave_span fields.',mmac_count_per_cu=counts,expected_mmac_instructions=expected,padded_tflops=n*bm*bn*k*2/time_us/1e6,useful_tflops=result['checked_elements']*k*2/time_us/1e6,cache=cache,cu_metrics=metrics)
(r/'model-metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
