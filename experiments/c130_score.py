import csv, numpy as np
R = list(csv.DictReader(open('c130_a/perturb.csv'))) + list(csv.DictReader(open('c130_b/perturb.csv'))); v = lambda t: 9000 if int(t) < 0 else int(t)
C83 = {38: 900, 23: 1200, 28: 1200, 35: 6000, 5: 5800, 37: 5750}
base = {int(r['s']): v(r['t4']) for r in R if float(r['sigma']) == 0}
print('sigma 0 reproduces C-83:', all(base[s] == C83[s] for s in C83), base)
for sig in (0.001, 0.01, 0.03, 0.1, 0.3):
    rr = [r for r in R if float(r['sigma']) == sig]
    lr = [np.log(v(r['t4']) / base[int(r['s'])]) for r in rr]
    fast = [v(r['t4']) / base[int(r['s'])] for r in rr if int(r['s']) in (38, 23, 28)]; slow = [base[int(r['s'])] / v(r['t4']) for r in rr if int(r['s']) in (35, 5, 37)]
    print(f'sigma {sig}: median |ln ratio| {np.median(np.abs(lr)):.2f} | fast: slower-by median {np.median(fast):.2f}, draws <=1.5x: {sum(x <= 1.5 for x in fast)}/9 | slow: faster-by median {np.median(slow):.2f}')
for s in (38, 23, 28, 35, 5, 37):
    print(s, 'base', base[s], ' '.join(f"{float(r['sigma'])}:{v(r['t4'])}" for r in R if int(r['s']) == s and float(r['sigma']) > 0))
