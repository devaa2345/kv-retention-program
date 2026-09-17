import sys, json, hashlib; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load; from kvre.task import build_prompt
from kvre.engine import Engine; from kvre.arms import make_cfg
model,tok=load(); eng=Engine(model,tok)
out={}
for arm in [1,2,3,4,5]:
    sig=[]
    for s in [9000,9001]:
        r=eng.run_prompt(build_prompt(s), make_cfg(arm,257), seed=s)
        sig.append(hashlib.sha256(json.dumps(r['retained_signature']).encode()).hexdigest()[:16])
        sig.append(round(r['accuracy'],6))
    out[f'arm{arm}']=sig
print(json.dumps(out))
