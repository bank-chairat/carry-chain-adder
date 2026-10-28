import csv, numpy as np, collections
D=collections.defaultdict(dict)
for r in csv.DictReader(open('c106/history.csv')): D[r['branch']][int(r['seed'])]=int(r['extra_steps_to_0.99_bit4'])
names=sorted(D); v=lambda t:3500 if t<0 else t
print('branch   '+' '.join(f'seed{s}' for s in range(5))+'  median')
for b in names: print(b,[v(D[b][s]) for s in range(5)],int(np.median([v(D[b][s]) for s in range(5)])))
c=lambda b,s:v(D[b][s])
p1=sum(c('cur100',s)>=1.5*c('cur000',s) for s in range(5))
p2=[sum(c(b,s)<1.5*c('cur000',s) for s in range(5)) for b in ('cur010','cur001')]
singles=['cur100','cur010','cur001']
p3=sum(abs(c('cur111',s)-max(c(b,s) for b in singles))<=0.25*max(c(b,s) for b in singles) for s in range(5))
print('P1',p1,'P2 (cur010,cur001)',p2,'P3',p3)
