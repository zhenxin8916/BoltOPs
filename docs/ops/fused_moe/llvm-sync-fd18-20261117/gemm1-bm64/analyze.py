"""CU0 active MMAC unions divided by CU0 own first-start/last-end lifetime."""
from pathlib import Path
import collections, configparser, json, re, sys
from perf_dispatch_span import read_dispatch_span

RX = re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)")

def events(path, starts, ends):
    current = {}
    for line in path.open():
        match = RX.match(line)
        if not match:
            continue
        tick, wid, iid, op = match.groups()
        w = int(wid, 16)
        if op.startswith("WAVE_START["):
            starts.append(int(tick))
        elif op.startswith("s_endpgm"):
            ends.append(int(tick))
        if "PC: 0x" in op:
            if w in current:
                yield w, current[w]
            current[w] = dict(tick=int(tick), op="", active=0, reads={})
        elif w in current:
            if op.startswith("r [S"):
                read = re.match(r"r \[S(\d+)\]([0-9a-f]+)", op)
                if read:current[w]["reads"].setdefault(int(read[1]), int(read[2],16))
            elif op.startswith("ACTIVE:"):
                current[w]["active"] = int(op.split()[-1], 16)
            elif op.startswith(("v_", "s_", "ds_", "buffer_", "matrix_", "global_", "flat_", "scratch_")):
                current[w]["op"] = op
    yield from current.items()

def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],b)
        else:
            merged.append([a,b])
    return merged

def analyze(d):
    launch=json.loads((d/"launch.json").read_text())
    result=json.loads((d/"result.json").read_text())
    assert result["mismatches"]==result["padded_row_errors"]==result["remaining_nan"]==result["untouched_modified"]==0
    assert result["checked_valid_elements"]==1580032
    cp=configparser.ConfigParser(strict=False);cp.read(d/"m5out/config.ini")
    period=int(cp[cp["system.shaders"]["clk_domain"]]["clock"])
    cu_sections=[s for s in cp.sections() if re.fullmatch(r"system.shaders.CUs\d+",s)]
    assert len(cu_sections)==48 and all(cp[s]["num_simds"]=="4" and cp[s]["wf_size"]=="64" for s in cu_sections)
    dispatch=read_dispatch_span(d)
    stats=(d/"m5out/0/0/stats.txt").read_text()
    def stat(key):return float(re.search(r"^"+re.escape(key)+r"\s+(\S+)",stats,re.M)[1])
    groups=[int(float(v)) for v in re.findall(r"^system.shaders.CUs\d+.completedWGs\s+(\S+)",stats,re.M)]
    assert len(groups)==48 and set(groups)=={1},groups
    assert stat("system.shaders.dispatcher.numKernelLaunched")==1
    mmops=[int(float(v)) for v in re.findall(r"^system.shaders.CUs\d+.numInstrTypeExecuted::MMOP\s+(\S+)",stats,re.M)]
    assert len(mmops)==48 and set(mmops)=={16512},mmops
    traces=list((d/"m5out/0/0").glob("itrace*.txt"))
    assert len(traces)==1 and traces[0].name=="itrace_cu0_se0_xcd0.txt",traces
    family="v_mmac_f32_16x16x32_fp8_fp8"
    inactive_mmac=collections.Counter();waits=collections.defaultdict(list);per_wave=collections.defaultdict(list);math_intervals=collections.defaultdict(list);intervals=collections.defaultdict(list);counts=collections.Counter();families=collections.Counter();starts=[];ends=[]
    for w,e in events(traces[0],starts,ends):
        if e["op"].startswith("s_abarrier_try_wait "):
            operand=e["op"].split(None,1)[1].split(",")[1].strip()
            waits[w].append(e["reads"][int(operand[1:])] if operand.startswith("s") else int(operand))
        if e["op"].startswith("v_mmac") and not e["active"]:
            inactive_mmac[e["op"].split()[0]]+=1
        if e["op"].startswith("v_mmac") and e["active"]:
            opcode=e["op"].split()[0];families[opcode]+=1
            intervals[w>>16].append((e["tick"],e["tick"]+8*period))
            if opcode != family:
                assert opcode=="v_mmac_f32_16x16x16_bf16",opcode
                continue
            per_wave[w].append(e["tick"])
            math_intervals[w>>16].append((e["tick"],e["tick"]+8*period));counts[str(w)]+=1
    assert len(counts)==8 and set(counts.values())=={2048},counts
    assert len(starts)==len(ends)==16,(len(starts),len(ends))
    simd_busy=[sum(b-a for a,b in union(intervals[i])) for i in range(4)]
    busy=sum(simd_busy); math_busy=sum(sum(b-a for a,b in union(math_intervals[i])) for i in range(4)); assert math_busy==16384*8*period
    cu_start=min(starts);cu_end=max(ends)+period;cu_span=cu_end-cu_start
    assert all(cu_start<=a<b<=cu_end for rows in intervals.values() for a,b in rows)
    assert len(waits)==16
    for wave, ids in waits.items():
        assert ids and ids[0]!=0 and all(0<=x<16 for x in ids), (wave, ids)
        assert all(a!=b for a,b in zip(ids,ids[1:])), (wave, ids)
    hot=[]
    for output in range(2):
        for simd in range(4):
            ticks=[t for w, rows in per_wave.items() if w>>16==simd for t in rows[output*1024:(output+1)*1024]]
            assert len(ticks)==2048
            hot.append(100*sum(b-a for a,b in union([(t,t+8*period) for t in ticks]))/(max(ticks)+8*period-min(ticks)))
    out=dict(raw_mmop_count_per_cu=mmops, fp8_math_instruction_count=sum(counts.values()), active_zero_bf16_instruction_count=families["v_mmac_f32_16x16x16_bf16"], math_mmac_total_pct=100*math_busy/(4*cu_span), inactive_mmac_counts=dict(inactive_mmac), abarrier_wait_ids={str(w):ids for w,ids in waits.items()}, hardware_wait_rules_passed=True, hot_loop_mmac_pct=sum(hot)/len(hot), per_output_simd_hot_pct=hot,time_us=dispatch["duration_us"],wave_span_time_us=cu_span/1e6,all_cu_wave_span_time_us=(stat("system.shaders.lastWaveEndTick")-stat("system.shaders.firstWaveStartTick"))/1e6,
             mmac_scope="cu0_wave_lifetime",mmac_total_pct=100*busy/(4*cu_span),cu0_first_wave_start_tick=cu_start,
             cu0_last_wave_end_tick=cu_end,cu0_lifetime_us=cu_span/1e6,simd_busy_ticks=simd_busy,
             dispatch_window_mmac_pct=100*busy/(4*dispatch["duration_ticks"]),clock_period_ps=period,
             effective_gc_ghz=stat("gcFrequency"),gc_ghz=1.8,
             active_mmac_families=dict(families),active_wave_counts=dict(counts),dispatch=dispatch,
             definition="CU0 four-SIMD active FP8 and BF16-zero MMAC 8-cycle interval unions / CU0 first WAVE_START to last s_endpgm plus one cycle; includes startup and tail; sb mmop-f16-i8-f8=8")
    (d/"metrics.json").write_text(json.dumps(out,indent=2)+"\n")
    print(d.name,out["time_us"],out["mmac_total_pct"],flush=True)
    return out

if __name__=="__main__":
    root=Path(__file__).resolve().parent
    for name in sys.argv[1:]:analyze(Path(name))
