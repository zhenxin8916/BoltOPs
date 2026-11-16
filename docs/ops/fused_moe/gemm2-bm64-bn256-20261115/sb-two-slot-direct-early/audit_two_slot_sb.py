from pathlib import Path
import re,json,sys,collections
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)");pending={};seq=collections.defaultdict(list)
def flush(w):
 if w in pending:seq[w&65535].append(pending[w])
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 for line in f:
  m=rx.match(line)
  if not m:continue
  t,w,ii,o=m.groups();t=int(t);w=int(w,16)
  if not 4 <= (w&65535)<16:continue
  if 'PC:' in o:
   flush(w);pending[w]=dict(tick=t,op='',active=0,s={},v={})
  elif w in pending:
   e=pending[w]
   if o.startswith('ACTIVE:'):e['active']=int(o.split()[-1],16)
   elif re.match(r'(s_|v_|ds_|buffer_|matrix_|global_)',o):e['op']=o
   elif o.startswith('r [S'):
    z=re.match(r'r \[S(\d+)\]([0-9a-f]+)',o)
    if z:e['s'][int(z[1])]=int(z[2],16)
   elif o.startswith('r [V'):
    z=re.match(r'r \[V(\d+)\]\[th00\]([0-9a-f]+)',o)
    if z:e['v'][int(z[1])]=int(z[2],16)
for w in list(pending):flush(w)
base=0x18800;writes={0:[],1:[]};consumer={};credits={}
for w,es in seq.items():
 if w<8:
  for e in es:
   if e['active'] and e['op'].startswith('buffer_load_') and ', lds' in e['op']:
    addr=e['s'][124];half=((addr-base)//512)%2
    writes[half].append(dict(tick=e['tick'],address=addr,wave=w))
 else:
  reads=[];cs=[]
  for e in es:
   if e['op'].startswith('ds_read2_b64'):
    z=re.match(r'ds_read2_b64 v(?:\[\d+:\d+\]|\d+), v(\d+)',e['op']);assert z,e['op']
    imm=re.search(r'offset:(\d+)',e['op']); encoded=int(imm[1]) if imm else 0
    first=(encoded&255)*8
    second=((encoded>>8)&255)*8
    assert second-first==256,e['op']
    reads.append(dict(tick=e['tick'],address=e['v'][int(z[1])]+first,base_address=e['v'][int(z[1])],encoded_offset=encoded))
   if e['op'].startswith('s_abarrier_arrive'):
    a=e['op'].split(None,1)[1].split(',');x=a[1].strip();bid=e['s'][int(x[1:])] if x.startswith('s') else int(x)
    if bid==(6 if w<12 else 8):cs.append(e['tick'])
  assert reads,(w,reads)
  cs=[t for t in cs if t>reads[0]['tick']];assert len(cs)==len(reads),(w,len(cs),len(reads))
  half=0 if w<12 else 1
  for gen,(read,credit) in enumerate(zip(reads,cs)):
   expected=base+(gen&1)*1024+half*512+((w%4)&1)*128
   assert read['address']==expected,(w,gen,read,hex(expected))
   assert any(read['tick']<e['tick']<credit and 'lgkmcnt(0)' in e['op'] for e in es),(w,gen,'no completion wait before credit')
  consumer[w]=reads;credits[w]=cs
assert len(consumer)==8
for half,ws in writes.items():
 ws.sort(key=lambda x:x['tick']);assert len(ws)==len(consumer[8+half*4]),(half,len(ws))
 for gen,e in enumerate(ws):
  assert e['address']==base+(gen&1)*1024+half*512,(half,gen,e)
  if gen>=2:
   last=max(credits[w][gen-2] for w in range(8+half*4,12+half*4))
   assert last<e['tick'],(half,gen,last,e)
result=dict(passed=True,scope='CU0 all four producer/consumer waves per role; actual SB destination M0 and consumer LDS base addresses, plus LGKM0 before B EMPTY and generation-2 credit ordering',base=base,slot_stride_bytes=1024,half_stride_bytes=512,producer_writes=writes,consumer_reads=consumer,consumer_empty_ticks=credits)
(r/'two-slot-sb-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(r.name,'PASS',[(w,len(v)) for w,v in consumer.items()])
