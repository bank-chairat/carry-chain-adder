import csv, sys, numpy as np
CAP={'1e-2':3000,'3e-3':3000,'1e-3':6000,'3e-4':12000}
for lr in ['3e-4','1e-3','3e-3','1e-2']:
    try: rows=list(csv.DictReader(open(f'c97_lr{lr}/gradshare_lr_n8.csv')))
    except Exception: continue
    J={};A={}
    for r in rows:
        t=int(r['t4']); s=int(r['seed'])
        (J if float(r['lam'])==1 else A)[s]=t
    ss=sorted(set(J)&set(A))
    def v(t): return CAP[lr] if t<0 else t
    jt=[J[s] for s in ss]; at=[A[s] for s in ss]
    rat=[v(J[s])/v(A[s]) for s in ss]
    print(lr,'n=',len(ss),'joint',jt,'alone',at,'ratios',[round(x,2) for x in rat],'median ratio',round(float(np.median(rat)),2),'median joint',np.median([v(x) for x in jt]),'median alone',np.median([v(x) for x in at]))
