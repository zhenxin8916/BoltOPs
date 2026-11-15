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
            current[w] = dict(tick=int(tick), op="", active=0)
        elif w in current:
            if op.startswith("ACTIVE:"):
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
    assert (d/"exit-code.txt").read_text().strip()=="0"
    launch=json.loads((d/"launch.json").read_text())
    result=json.loads((d/"result.json").read_text())
    assert result["verified"] and result["mismatches"]==0 and result["untouched_tail"]
    assert result["checked_elements"]==4380672
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
    assert len(mmops)==48 and set(mmops)=={4096},mmops
    traces=list((d/"m5out/0/0").glob("itrace*.txt"))
    assert len(traces)==1 and traces[0].name=="itrace_cu0_se0_xcd0.txt",traces
    family="v_mmac_i32_16x16x32_i8" if launch["kind"]=="int8" else "v_mmac_f32_16x16x32_fp8_fp8"
    intervals=collections.defaultdict(list);counts=collections.Counter();families=collections.Counter();starts=[];ends=[]
    for w,e in events(traces[0],starts,ends):
        if e["op"].startswith("v_mmac") and e["active"]:
            opcode=e["op"].split()[0];families[opcode]+=1
            assert opcode==family,opcode
            intervals[w>>16].append((e["tick"],e["tick"]+8*period));counts[str(w)]+=1
    assert len(counts)==8 and set(counts.values())=={512},counts
    assert len(starts)==len(ends)==16,(len(starts),len(ends))
    simd_busy=[sum(b-a for a,b in union(intervals[i])) for i in range(4)]
    busy=sum(simd_busy);assert busy==4096*8*period
    cu_start=min(starts);cu_end=max(ends)+period;cu_span=cu_end-cu_start
    assert all(cu_start<=a<b<=cu_end for rows in intervals.values() for a,b in rows)
    out=dict(time_us=dispatch["duration_us"],wave_span_time_us=(stat("system.shaders.lastWaveEndTick")-stat("system.shaders.firstWaveStartTick"))/1e6,
             mmac_scope="cu0_wave_lifetime",mmac_total_pct=100*busy/(4*cu_span),cu0_first_wave_start_tick=cu_start,
             cu0_last_wave_end_tick=cu_end,cu0_lifetime_us=cu_span/1e6,simd_busy_ticks=simd_busy,
             dispatch_window_mmac_pct=100*busy/(4*dispatch["duration_ticks"]),clock_period_ps=period,
             effective_gc_ghz=stat("gcFrequency"),gc_ghz=int(d.name.split("gc")[1])/1000,
             active_mmac_families=dict(families),active_wave_counts=dict(counts),dispatch=dispatch,
             definition="CU0 four-SIMD active MMAC 8-cycle interval unions / CU0 first WAVE_START to last s_endpgm plus one cycle; includes startup and tail")
    (d/"metrics.json").write_text(json.dumps(out,indent=2)+"\n")
    print(d.name,out["time_us"],out["mmac_total_pct"],flush=True)
    return out

if __name__=="__main__":
    root=Path(__file__).resolve().parent
    for name in sys.argv[1:]:analyze(root/name)
