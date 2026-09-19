import json, random, collections
from p3.natural import score as S
ds={}
for l in open('data/natural/nat_v1.jsonl'):
    d=json.loads(l); ds[d['instance_id']]=d
rows=[json.loads(l) for l in open('runs/nvidia/nat_gate_nat_v1_M3.jsonl')]
Z=[]
for r in rows:
    if r['arm']!='full_cache': continue
    P=ds[r['iid']]['persons']
    for lv,d in r['levels'].items():
        for k,(g,a) in enumerate(zip(d['gen'],d['answers'])):
            if S.score_one(g,a,P)==0.0:
                gn=S.norm(g)
                allel=all(S.contains(gn,e) for e in a)
                oth=[p for p in P if S.norm(p)!=S.norm(a[0]) and S.contains(gn,p)]
                Z.append(dict(iid=r['iid'],lv=lv,k=k,gen=g,gold=a,all_elements_present=allel,others=oth))
c=collections.Counter((z['lv'],z['all_elements_present']) for z in Z)
print('zeros by level, all_gold_elements_present:',dict(c)); print('total',len(Z),collections.Counter(z['lv'] for z in Z))
random.Random(7).shuffle(Z)
out=[];seen=collections.Counter()
for z in Z:
    if seen[z['lv']]<16: seen[z['lv']]+=1; out.append(z)
json.dump(out,open('out/nat_audit_sample.json','w'),indent=1)
for z in sorted(out,key=lambda z:z['lv']):
    print('---L%s %s present=%s others=%s\n GEN=%r\n GOLD=%s'%(z['lv'],z['iid'],z['all_elements_present'],z['others'],z['gen'][:260],z['gold']))
