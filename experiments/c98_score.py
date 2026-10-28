import csv, numpy as np
base=[r for r in csv.DictReader(open('c39_W128_lr1e-3/wide_n8.csv'))]
def mx(rows):
    out=[]
    for r in rows:
        t=[int(r[f't{b}']) for b in range(9)]
        out.append(10000 if min(t)<0 else max(t))
    return out
cells=[('1 (C-39 control)',base)]+[(a,list(csv.DictReader(open(f'c98_a{a}/wide_n8.csv')))) for a in ('0.5','2','4','8')]
form={'0.5':4500,'2':3100,'4':2050,'8':1300,'1 (C-39 control)':4000}
for a,rows in cells:
    m=mx(rows); print('alpha',a,'max-over-bits per seed',m,'median',int(np.median(m)),'formula',form[a],'ratio to control',round(np.median(m)/4000,2), 'bit4 per seed',[int(r['t4']) for r in rows])
