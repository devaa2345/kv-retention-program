import json, collections
from p3.natural import score as S
ds={}
for l in open('data/natural/nat_v1.jsonl'):
    d=json.loads(l); ds[d['instance_id']]=d
for tag in ('M2','M3'):
  for arm in ('full_cache','floor_pos'):
    rows=[json.loads(l) for l in open('runs/nvidia/nat_gate_nat_v1_%s.jsonl'%tag)]
    C=collections.Counter(); tot=collections.Counter()
    for r in rows:
        if r['arm']!=arm: continue
        P=ds[r['iid']]['persons']
        for lv,d in r['levels'].items():
            for g,a,fl in zip(d['gen'],d['answers'],d['flags']):
                tot[lv]+=1
                if S.score_one(g,a,P)==1.0: continue
                gn=S.norm(g)
                pres=[S.contains(gn,e) for e in a]
                oth=any(S.norm(p)!=S.norm(a[0]) and S.contains(gn,p) for p in P)
                if all(pres): k='A_scorer: correct + continuation naming other person'
                elif pres[0] and not oth and fl=='cap': k='B_cap: correct record, cut off by max_new'
                elif pres[0]: k='C_partial: right person, wrong/missing field (not cap)' if not oth else 'C2_partial+other person'
                else: k='D_real: wrong/absent person'
                C[(lv,k)]+=1
    print('\n',tag,arm,'variants',dict(tot))
    for k,v in sorted(C.items()): print('  L%s %-62s %d'%(k[0],k[1],v))
