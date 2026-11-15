import json,re,sys,collections
from pathlib import Path
r=Path(sys.argv[1]); result=json.loads((r/'result.json').read_text());assert result['mismatches']==0
manifest=json.loads((r/'address-manifest.json').read_text());page=1<<21;vm={};va=None
log=r/'run.log'
with log.open() as f:
 for line in f:
  m=re.search(r'Starting page table walk for VA: 0x([0-9a-f]+)',line)
  if m:va=int(m[1],16)
  m=re.search(r'contents=0x([0-9a-f]{16}).*big page detected at PDE0',line)
  if m:
   assert va is not None
   v=va&-page;p=int(m[1],16)&((1<<48)-page)
   assert v not in vm or vm[v]==p,(hex(v),hex(p),hex(vm[v]))
   vm[v]=p
ranges=[]
for name,item in manifest.items():
 begin=item['address'];end=begin+item['bytes']
 for v in range(begin&-page,(end+page-1)&-page,page):
  if v in vm:ranges.append((vm[v]+max(begin-v,0),vm[v]+min(end-v,page),name))
requests={};native=[];hist=collections.Counter();sizes=collections.Counter();mods=collections.Counter();unique=collections.defaultdict(set)
with log.open() as f:
 for line in f:
  m=re.search(r'UMC_REQUEST cmd=(\d+) id=(\d+) addr=([0-9a-f]+) size=(\d+) flags=([0-9a-f]+)',line)
  if m:
   cmd,ident=int(m[1]),int(m[2]);addr,size=int(m[3],16),int(m[4]);assert ident not in requests
   matches=[n for lo,hi,n in ranges if lo<=addr<hi];assert len(matches)<=1
   name=matches[0] if matches else ('page_tables' if addr>=1<<36 else 'other')
   requests[ident]=(cmd,name,addr,size);hist[(name,cmd)]+=1;sizes[(name,cmd,size)]+=1;mods[(cmd,addr%256)]+=1;unique[(name,cmd)].add(addr//256)
  m=re.search(r'send response od pkt: (\d+)',line)
  if m:native.append(int(m[1]))
assert len(native)==len(set(native)) and set(native)==set(requests),(len(native),len(requests))
s=(r/'m5out/0/0/stats.txt').read_text()
def val(suffix):
 m=re.search(r'^system.umcs\.'+suffix+r'\s+(\S+)',s,re.M)
 if m:return int(float(m[1]))
# Do not infer payload bandwidth from packet.size: data reads may carry only
# the originating 32/64-byte request, while writeback packets are full lines.
out=dict(diagnostic_only=True,numerical_errors=0,checked_elements=result['checked_elements'],mapping_source='Native UTCL2_PAGETABLE PDE0 2MiB mappings; manifest GPU virtual tensor ranges',mapped_pages=len(vm),admissions=len(native),requests=[dict(tensor=n,cmd=c,count=count,unique_256B_lines=len(unique[(n,c)]),repeat_256B_line_requests=count-len(unique[(n,c)])) for (n,c),count in sorted(hist.items())],packet_size_histogram=[dict(tensor=n,cmd=c,size_bytes=z,count=count) for (n,c,z),count in sorted(sizes.items())],address_mod_256_histogram=[dict(cmd=c,offset=o,count=count) for (c,o),count in sorted(mods.items())],scope_note='Read-only audit changes allocation/packet IDs and is diagnostic, not a formal time comparison. Packet size is not asserted to be actual UMC bus payload. Unknown address ranges are retained explicitly.')
(r/'umc-address-classification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
