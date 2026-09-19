import json, statistics as st
from p3.natural import score as S
ds={json.loads(l)['instance_id']:json.loads(l) for l in open('data/natural/nat_v1.jsonl')}
def old(g,e,P):  # v1 rule, for comparison
    g=S.norm(g)
    if not all(S.contains(g,x) for x in e): return 0.0
    return 0.0 if any(S.norm(p)!=S.norm(e[0]) and S.contains(g,p) for p in P) else 1.0
for tag in ('M2','M3'):
    rows=[json.loads(l) for l in open('runs/nvidia/nat_gate_nat_v1_%s.jsonl'%tag)]
    for arm,C in (('full_cache',0),('floor_pos',256),('floor_pos',512),('floor_pos',1024),('full_cache_deleted',0)):
        for lv in '135':
            a=[];b=[]
            for r in rows:
                if r['arm']!=arm or r['C']!=C: continue
                P=ds[r['iid']]['persons']; d=r['levels'][lv]
                a.append(st.fmean(old(g,e,P) for g,e in zip(d['gen'],d['answers'])))
                b.append(st.fmean(S.score_one(g,e,P) for g,e in zip(d['gen'],d['answers'])))
            print(tag,arm,C,'L'+lv,'v1 %.3f  v2 %.3f'%(st.fmean(a),st.fmean(b)))
