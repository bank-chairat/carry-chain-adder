import csv, numpy as np, collections
D=collections.defaultdict(lambda:[0,0])
for r in csv.DictReader(open('c100/strata.csv')):
    k=(int(r['seed']),int(r['step']),int(r['bit']),int(r['L'])); D[k]=[int(r['count']),int(r['errors'])]
def agg(s,t,b):
    c=sum(D[(s,t,b,l)][0] for l in range(b+1)); e=sum(D[(s,t,b,l)][1] for l in range(b+1)); return e/c
def er(s,t,b,l): c,e=D[(s,t,b,l)]; return e/c if c else float('nan')
print("== Prediction 1: bit 4 first eval with aggregate error in [0.2,0.5]; error in L=0 stratum")
n1=0
for s in range(5):
    t=next((t for t in range(0,5001,50) if 0.2<=agg(s,t,4)<=0.5),None)
    if t is None: print(s,'none'); continue
    print('seed',s,'step',t,'agg',round(agg(s,t,4),3),'by L',[round(er(s,t,4,l),3) for l in range(5)]); n1+= er(s,t,4,0)>=0.15
print('seeds with L=0 error >=0.15:',n1)
print("== Prediction 2/3: first eval with aggregate accuracy >=0.99 (error<=0.01)")
for b in (3,4,5,6):
    rat=[];
    for s in range(5):
        t=next((t for t in range(0,5001,50) if agg(s,t,b)<=0.01),None)
        if t is None: print('bit',b,'seed',s,'never'); continue
        e0=er(s,t,b,0); em=er(s,t,b,b); rat.append(em/e0 if e0>0 else np.inf)
        print('bit',b,'seed',s,'step',t,'agg',round(agg(s,t,b),4),'by L',[round(er(s,t,b,l),4) for l in range(b+1)],'ratio Lmax/L0',round(rat[-1],1) if np.isfinite(rat[-1]) else 'inf (L0 err 0)')
    print('bit',b,'median ratio',np.median(rat),'n>=3:',sum(r>=3 for r in rat),'n>=2:',sum(r>=2 for r in rat))
