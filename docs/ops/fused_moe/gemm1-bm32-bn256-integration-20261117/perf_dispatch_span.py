"""Read the single-dispatch XCD time section from PMD HYGPERF.

Layout verified against b61a79c5 PerfWriter::OutputXCDTimeSection:
section type 15, 72 bytes; offset is relative to the 24-byte file header.
The final four uint64s are begin/end ticks and begin/end nanoseconds.
"""
import hashlib, struct
from pathlib import Path

def read_dispatch_span(case):
    root = Path(case)
    paths = list((root / 'm5out/0/0').glob('*_python*.perf'))
    assert len(paths) == 1, paths
    path = paths[0]
    data = path.read_bytes()
    assert data[:8] == b'\xe4HYGPERF', data[:8]
    header_size = struct.unpack_from('<Q', data, 16)[0]
    assert header_size == 24, header_size
    matches = []
    pos = header_size
    marker = struct.pack('<Q', 15)
    while True:
        off = data.find(marker, pos)
        if off < 0:
            break
        pos = off + 1
        if off + 72 <= len(data):
            q = struct.unpack_from('<9Q', data, off)
            if q[1] == off - header_size and q[2] == 72:
                matches.append((off, q))
    assert len(matches) == 1, matches
    off, q = matches[0]
    begin, end, begin_ns, end_ns = q[5:9]
    assert 0 <= begin < end and 0 <= begin_ns < end_ns
    assert abs(begin - begin_ns * 1000) < 2000
    assert abs(end - end_ns * 1000) < 2000
    return dict(begin_tick=begin, end_tick=end, duration_ticks=end-begin,
                duration_us=(end-begin)/1e6, gui_duration_us=(end_ns-begin_ns)/1000,
                section_offset=off, section_type=15, source_perf=str(path),
                source_perf_sha256=hashlib.sha256(data).hexdigest())
