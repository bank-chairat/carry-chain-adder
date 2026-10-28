import csv, numpy as np
from scipy.stats import spearmanr
T = {}
for d in ('c83_a', 'c83_b'):
    for r in csv.DictReader(open(f'{d}/init_align_joint_n8.csv')): T[int(r['s'])] = int(r['time'])
for s in range(40, 60): T[s] = int(np.load(f'c127/traj_s{s}.npz')['t4'])
TA = {}
for d in ('c82_a', 'c82_b'):
    for r in csv.DictReader(open(f'{d}/init_align_n8.csv')): TA[int(r['s'])] = int(r['time'])
def M(r):
    xs, ys = np.log([2, 4, 8]), np.log([1.0, 2.0, 3.6])
    if r <= 2: return 1.0
    return float(min(np.exp(np.interp(np.log(r), xs, ys, right=np.log(3.6) + (np.log(r) - np.log(8)) * (np.log(3.6) - np.log(2.0)) / (np.log(8) - np.log(4)))), 4.2))
rows = []
for s in range(60):
    z = np.load(f'c139/init_s{s}.npz'); sh = float(np.mean(z['share']))      # steps 25..500
    ta = TA.get(s, 250); r = (1 - sh) / sh; That = ta / sh * M(r)
    rows.append((s, T[s], That, sh, r, z['share'][:4].mean()))
t = np.array([x[1] for x in rows], float); th = np.array([x[2] for x in rows]); sh = np.array([x[3] for x in rows]); early = np.array([x[5] for x in rows])
print('share s (steps 25-500): median %.3f range %.3f-%.3f; implied r median %.1f' % (np.median(sh), sh.min(), sh.max(), np.median([x[4] for x in rows])))
print('LAW: median |ln(T_hat/T)| %.2f; Spearman(T_hat, T) %.2f; median T_hat %.0f vs median T %.0f' % (np.median(np.abs(np.log(th / t))), spearmanr(th, t).statistic, np.median(th), np.median(t)))
print('Spearman(share 25-500, T) %.2f; Spearman(share steps 25-100 only, T) %.2f' % (spearmanr(sh, t).statistic, spearmanr(early, t).statistic))
