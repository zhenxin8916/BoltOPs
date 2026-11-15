import collections, configparser, json, re, sys
from pathlib import Path
from perf_dispatch_span import read_dispatch_span
r = Path(sys.argv[1])
result = json.loads((r / 'result.json').read_text())
assert result['mismatches'] == 0
launch = json.loads((r / 'launch.json').read_text())
config = configparser.ConfigParser()
config.read(r / 'm5out/config.ini')
s = (r / 'm5out/0/0/stats.txt').read_text()
def stat(key):
    m = re.search(r'^' + re.escape(key) + r'\s+(\S+)', s, re.M)
    assert m, key
    return float(m[1])
cu_config=[sec for sec in config.sections() if re.fullmatch(r'system\.shaders\.CUs\d+',sec)]
assert len(cu_config)==48
assert all(int(config[sec]['num_simds'])==4 and int(config[sec]['wf_size'])==64 for sec in cu_config)
gc_ps = int(config[config['system.shaders']['clk_domain']]['clock'])
umc_ps = int(config[config['system.umcs']['clk_domain']]['clock'])
start, end = [stat('system.shaders.' + k) for k in ['firstWaveStartTick', 'lastWaveEndTick']]
rx = re.compile(r'(\d+): system.umcs: send response od pkt: (\d+) to noc')
responses = []
with (r / 'run.log').open(errors='replace') as f:
    for line in f:
        m = rx.search(line)
        if m:
            responses.append((int(m[1]), int(m[2])))
reads, writes = [int(stat('system.umcs.num' + k + 'Requests')) for k in ['Read', 'Write']]
assert len(responses) == reads + writes, (len(responses), reads, writes)
assert len({pkt for t, pkt in responses}) == len(responses), 'duplicate UMC response ids'
counts = collections.Counter(t for t, pkt in responses)
groups = [sec for sec in config.sections() if config[sec].get('type') == 'TccBankGroup']
assert config['system.umcs']['rw_credit_share'] == 'true'
ports = len(config['system.umcs']['resp_ports'].split())
assert ports == 2 * len(groups)
assert max(counts.values()) <= ports
# Shared credits do not mean one response per group per cycle. Bound the
# complete fixed-latency residence window instead.
residence_cycles = int(config['system.umcs']['umc_latency'])
credit_per_group = int(config['system.umcs']['umc_credit'])
q = collections.deque()
rolling = peak_rolling = 0
for t, count in sorted(counts.items()):
    q.append((t, count))
    rolling += count
    while t - q[0][0] >= residence_cycles * umc_ps:
        rolling -= q.popleft()[1]
    peak_rolling = max(peak_rolling, rolling)
assert peak_rolling <= len(groups) * credit_per_group
period_phases = {t % umc_ps for t, pkt in responses}
assert len(period_phases) == 1, period_phases
families = set(re.findall(r'\b(v_mmac_\w+)', (r / 'gemm/kernel.amdgcn').read_text()))
assert families == {'v_mmac_f32_16x16x32_fp8_fp8'}, families
mmac = {int(cu): int(n) for cu, n in re.findall(r'^system.shaders.CUs(\d+)\.numInstrTypeExecuted::MMOP\s+(\d+)', s, re.M)}
assert len(mmac) == 48
ops = launch['measured_logical_tiles'] * launch['options']['BLOCK_SIZE_M'] * launch['options']['BLOCK_SIZE_N'] * launch['actual_K'] * 2
assert sum(mmac.values()) * 16384 == ops
wave_span_time_us=(end-start)/1e6
wave_span_mmac_percent=sum(mmac.values())*8*gc_ps/(48*4*(end-start))*100
span=read_dispatch_span(r)
assert stat('system.shaders.dispatcher.numKernelLaunched')==1
assert abs(span['duration_ticks']-stat('totalSimTicks'))<=max(1000,gc_ps)
start,end=span['begin_tick'],span['end_tick']
whole = sum(mmac.values()) * 8 * gc_ps / (48 * 4 * (end-start)) * 100
during = sum(start <= t <= end for t,p in responses)
out = dict(schema_version=2,grid=launch['grid'], diagnostic_only=launch['diagnostic_only'],dispatch_span=span,
           wave_span_time_us=wave_span_time_us,wave_span_mmac_percent=wave_span_mmac_percent,
           time_us=(end-start)/1e6, whole_48cu_mmac_percent=whole,
           deadline_for_40_percent_us=sum(mmac.values())*8*gc_ps/(48*4*.4)/1e6,
           numerical_errors=result['mismatches'], read_requests=reads, write_requests=writes,
           logged_umc_admissions=len(responses), admissions_before_dispatch=sum(t<start for t,p in responses),
           admissions_during_dispatch=sum(start<=t<=end for t,p in responses),
           admissions_after_dispatch=sum(t>end for t,p in responses),
           last_admission_minus_dispatch_end_us=(max(counts)-end)/1e6,
           umc_period_ps=umc_ps, bank_groups=len(groups), response_ports=ports, max_admissions_per_tick=max(counts.values()),
           umc_latency_cycles=residence_cycles, credit_per_group=credit_per_group,
           peak_admissions_in_residence_window=peak_rolling,
           combined_credit_pool=len(groups)*credit_per_group,
           credit_ceiling_transfers_per_us=len(groups)*credit_per_group/(residence_cycles*umc_ps/1e6),
           admission_phase_ps=list(period_phases),
           admission_port_slot_utilization_percent=100*during*umc_ps/(ports*(end-start)),
           combined_credit_residence_utilization_percent=100*during*residence_cycles*umc_ps/(len(groups)*credit_per_group*(end-start)),
           definition='The native log says send response, but the verified UmcSimulate::sendResp control flow emits it when admitting a request into the latency queue, before actual sendResp at maturity. These are UMC admission timestamps, not completion or credit-return times. Dispatch uses the complete HYGPERF XCD time section, including startup; wave_span fields preserve the narrower historical interval. Counts match complete model R+W counters; all-CU MMOP geometry/work checked. Port-slot occupancy is not payload bandwidth.',
           traffic_at_256B_per_transaction_MiB=len(responses)*256/2**20)
kind_rx = re.compile(r'UMC_REQUEST cmd=(\d+) id=(\d+)')
kinds = {}
with (r / 'run.log').open(errors='replace') as f:
    for line in f:
        m = kind_rx.search(line)
        if m:
            cmd, pkt = map(int, m.groups())
            assert pkt not in kinds
            kinds[pkt] = cmd
if kinds:
    assert set(kinds) == {pkt for t,pkt in responses}
    kc = collections.Counter(kinds.values())
    assert set(kc) <= {1,2,72}, kc
    assert kc[1] + kc[2] == reads and kc[72] == writes
    kr = collections.Counter(t for t,pkt in responses if kinds[pkt] in (1,2))
    kw = collections.Counter(t for t,pkt in responses if kinds[pkt] == 72)
    assert max(kr.values()) <= len(groups) and max(kw.values()) <= len(groups)
    out['request_kind_audit'] = dict(command_counts=dict(kc), request_ids_equal_native_log_ids=True,
                                   max_read_admissions_per_tick=max(kr.values()),
                                   max_write_admissions_per_tick=max(kw.values()))
(r / 'umc-response-window.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
