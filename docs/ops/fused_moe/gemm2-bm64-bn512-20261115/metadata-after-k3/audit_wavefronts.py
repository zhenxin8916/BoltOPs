"""Validate nonempty PMD SQTT sections before delivering a Wavefronts perf."""
import hashlib,json,struct,sys
from pathlib import Path

def audit(path):
    path=Path(path);data=path.read_bytes()
    assert data[:8]==b"\xe4HYGPERF" and struct.unpack_from("<Q",data,16)[0]==24
    sections=[];pos=24;marker=struct.pack("<Q",11)
    while True:
        off=data.find(marker,pos)
        if off<0:break
        pos=off+1
        if off+80>len(data):continue
        q=struct.unpack_from("<10Q",data,off)
        if q[1]!=off-24 or q[2]!=80:continue
        assert q[6]==off-24+80
        assert q[7]>0 and q[8]>0 and q[6]+24+q[8]<=len(data)
        sections.append(dict(offset=off,raw_bytes=q[7],compressed_bytes=q[8]))
    assert sections,"No nonempty SQTT section: enable TT0 with Perf, not IT0 alone"
    return dict(perf_path=str(path),perf_sha256=hashlib.sha256(data).hexdigest(),perf_bytes=len(data),sqtt_sections=sections,nonempty_sqtt_verified=True)

if __name__=="__main__":
    case=Path(sys.argv[1]);paths=list((case/"m5out/0/0").glob("*python*.perf"));assert len(paths)==1
    result=audit(paths[0]);(case/"wavefronts-audit.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
