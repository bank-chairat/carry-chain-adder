import csv, numpy as np
from scipy.stats import spearmanr
R=[]
for d in ('c111_a','c111_b'): R+=list(csv.DictReader(open(f'{d}/diag.csv')))
conds=['none_K0','all9_K100','all9_K250','all9_K500','all9_K1000','others8_K250','random9_K250']
keys=[k for k in R[0] if k not in ('seed','cond','cost')]
print('median per state (cost and measures)')
print('state'.ljust(14),'cost',' '.join(k[:14].rjust(14) for k in ['L1_dead','L2_dead','L3_dead','L2_erank','L3_erank','L2_pr','L2_normratio','L2_actfrac','L1_irr_over_rel']))
for c in conds:
    rr=[r for r in R if r['cond']==c]; med=lambda k: np.median([float(r[k]) for r in rr])
    print(c.ljust(14),str(int(med('cost'))).rjust(4),' '.join(f"{med(k):14.3f}" for k in ['L1_dead','L2_dead','L3_dead','L2_erank','L3_erank','L2_pr','L2_normratio','L2_actfrac','L1_irr_over_rel']))
print('per-seed costs:'); 
for c in conds: print(c, [int([r for r in R if r['cond']==c and int(r['seed'])==s][0]['cost']) for s in range(5)])
cost=[float(r['cost']) for r in R]
print('max dead fraction any layer/state:', max(max(float(r['L1_dead']),float(r['L2_dead']),float(r['L3_dead'])) for r in R))
print('Spearman with cost over 35 points:')
for k in keys:
    rho=spearmanr([float(r[k]) for r in R],cost).statistic; print(f'  {k:16s} {rho:+.2f}')
# excluding the solved all9_K1000 seed2 (cost 100 because bit 4 was nearly solved)
