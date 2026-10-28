import csv, numpy as np
from scipy.stats import spearmanr, rankdata
T = {}
for d in ('c83_a', 'c83_b'):
    for r in csv.DictReader(open(f'{d}/init_align_joint_n8.csv')): T[int(r['s'])] = int(r['time'])
for s in range(40, 60): T[s] = int(np.load(f'c127/traj_s{s}.npz')['t4'])
S = list(range(60)) + list(range(100, 500)); D = {s: np.load(f'c139/init_s{s}.npz') for s in S}
def gram(s):
    W = np.c_[D[s]['W1'], D[s]['b1']].astype(np.float64); M = W.T @ W; iu = np.triu_indices(17); return M[iu]
J500 = {s: float(D[s]['J'][-1]) for s in S}; share = {s: float(D[s]['share'].mean()) for s in S}; share_early = {s: float(D[s]['share'][:4].mean()) for s in S}
allS = S
print('(1) Spearman(share 25-500, J500) over %d inits: %.2f; early-share (25-100) vs J500: %.2f' % (len(allS), spearmanr([share[s] for s in allS], [J500[s] for s in allS]).statistic, spearmanr([share_early[s] for s in allS], [J500[s] for s in allS]).statistic))
k = list(range(60)); print('(2) on the 60 with T: Spearman(share 25-500, T) %.2f; J500 vs T %.2f' % (spearmanr([share[s] for s in k], [T[s] for s in k]).statistic, spearmanr([J500[s] for s in k], [T[s] for s in k]).statistic))
tr = list(range(100, 400)); va = list(range(400, 500))
Xtr = np.array([gram(s) for s in tr]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12; Z = lambda ss: (np.array([gram(s) for s in ss]) - mu) / sd
ytr = rankdata([J500[s] for s in tr]) / len(tr); ytr = ytr - ytr.mean()
def ridge(Xa, ya, lam): return np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya)
Ztr = Z(tr); best = None
for lam in (1, 10, 100, 1000, 10000, 1e5):
    sc = []
    for f in range(5):
        te_i = np.arange(len(tr)) % 5 == f; b = ridge(Ztr[~te_i], ytr[~te_i], lam); sc.append(spearmanr(Ztr[te_i] @ b, ytr[te_i]).statistic)
    print('  lambda %g: 5-fold CV Spearman %.3f' % (lam, np.mean(sc)))
    if best is None or np.mean(sc) > best[1]: best = (lam, np.mean(sc))
b = ridge(Ztr, ytr, best[0])
print('(3) ridge on Gram features, lambda %g: held-out (s 400-499) Spearman with J500 %.2f; on s 0-59: Spearman(prediction, T) %.2f' % (best[0], spearmanr(Z(va) @ b, [J500[s] for s in va]).statistic, spearmanr(Z(k) @ b, [T[s] for s in k]).statistic))
names = []
for i in range(17):
    for j in range(i, 17): names.append(f"{'a%d' % i if i < 8 else ('b%d' % (i - 8) if i < 16 else 'bias')}*{'a%d' % j if j < 8 else ('b%d' % (j - 8) if j < 16 else 'bias')}")
top = np.argsort(-np.abs(b))[:12]; print('largest coefficients:', ', '.join(f'{names[t]} {b[t]:+.3f}' for t in top))
np.save('c139_ridge_coef.npy', b)
