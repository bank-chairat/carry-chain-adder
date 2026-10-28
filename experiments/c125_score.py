import csv, glob, numpy as np
from scipy.stats import spearmanr
T={}
for d in ('c83_a','c83_b'):
    for r in csv.DictReader(open(f'{d}/init_align_joint_n8.csv')): T[int(r['s'])]=int(r['time'])
rows=[]
for s in range(40):
    z=np.load(f'c125/traj_s{s}.npz'); st, c, t4 = z['steps'], z['corr'], int(z['t4'])
    on=None
    for k in range(len(c)-2):
        if c[k]>=0.05 and c[k+1]>=0.05 and c[k+2]>=0.05: on=int(st[k]); break
    # exponential fit of log corr between 0.05 and 0.3 (first crossing segment after onset)
    r2=slope=np.nan
    if on is not None:
        k0=list(st).index(on); idx=[k for k in range(k0,len(c)) if 0.05<=c[k]<=0.3]
        # keep the first contiguous block
        blk=[]
        for k in idx:
            if not blk or k==blk[-1]+1: blk.append(k)
            else: break
        if len(blk)>=3:
            x=st[blk].astype(float); y=np.log(c[blk]); A=np.vstack([x,np.ones_like(x)]).T; coef,res,_,_=np.linalg.lstsq(A,y,rcond=None); pred=A@coef
            r2=1-((y-pred)**2).sum()/max(((y-y.mean())**2).sum(),1e-12); slope=coef[0]
    rows.append((s,t4,T[s],on,(t4-on) if on is not None else None,r2,slope))
same=sum(r[1]==r[2] for r in rows); print('reproduce C-83:',same,'of 40')
ok=[r for r in rows if r[3] is not None]
Tt=np.array([r[1] for r in ok],float); on=np.array([r[3] for r in ok],float); D=Tt-on
cv=lambda x: x.std()/x.mean()
print('runs with an onset:',len(ok),'| CV of T %.2f, CV of D %.2f | Spearman(t_on,T) %.2f'%(cv(Tt),cv(D),spearmanr(on,Tt).statistic))
print('t_on: median %.0f range %.0f-%.0f | D: median %.0f range %.0f-%.0f | T: median %.0f range %.0f-%.0f'%(np.median(on),on.min(),on.max(),np.median(D),D.min(),D.max(),np.median(Tt),Tt.min(),Tt.max()))
r2s=[r[5] for r in ok if not np.isnan(r[5])]; sl=[r[6] for r in ok if not np.isnan(r[6])]
print('exponential fit in 0.05-0.3: runs with >=3 points %d, median R2 %.2f, growth rate median %.4f per step (CV %.2f), e-folding %.0f steps'%(len(r2s),np.median(r2s),np.median(sl),np.std(sl)/np.mean(sl),1/np.median(sl)))
print('Spearman(D, T) %.2f; Spearman(growth rate, T) %.2f'%(spearmanr(D,Tt).statistic, spearmanr([r[6] for r in ok if not np.isnan(r[6])],[r[1] for r in ok if not np.isnan(r[6])]).statistic))
for r in sorted(rows,key=lambda r:r[1])[::5]: print('  s',r[0],'T',r[1],'t_on',r[3],'D',r[4],'R2 %.2f'%r[5] if not np.isnan(r[5]) else '', 'rate %.4f'%r[6] if not np.isnan(r[6]) else '')
import json; json.dump([dict(s=r[0],T=r[1],t_on=r[3],D=r[4]) for r in rows],open('c125_onsets.json','w'))
