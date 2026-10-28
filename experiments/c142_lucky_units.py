import csv, numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
torch.manual_seed(0)
T = {}
for d in ('c83_a', 'c83_b'):
    for r in csv.DictReader(open(f'{d}/init_align_joint_n8.csv')): T[int(r['s'])] = int(r['time'])
for s in range(40, 60): T[s] = int(np.load(f'c127/traj_s{s}.npz')['t4'])
S = list(range(60)) + list(range(100, 500)); D = {s: np.load(f'c139/init_s{s}.npz') for s in S}
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float64)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel(); X = np.concatenate([bits(A, 8), bits(B, 8)], 1)
def feats(s):
    W = D[s]['W1'].astype(np.float64); b = D[s]['b1'].astype(np.float64); p = ((X @ W.T + b) > 0).mean(0)
    prod = W[:, :8] * W[:, 8:]; diff = np.abs(W[:, :8] - W[:, 8:])
    return np.c_[W, b, W ** 2, prod, diff, p, p * (1 - p)], W, b, p
F = {}; HAND = {}
for s in S:
    f, W, b, p = feats(s); F[s] = f
    k = (p >= 0.2) & (p <= 0.8); ratio = (W[:, 4] ** 2 + W[:, 12] ** 2) / (W[:, 7] ** 2 + W[:, 15] ** 2 + 1e-12); HAND[s] = float(ratio[k].max()) if k.any() else 0.0
J = {s: float(D[s]['J'][-1]) for s in S}; lab = {s: 1.0 if J[s] >= 0.2 else (0.0 if J[s] < 0.05 else None) for s in S}
tr = [s for s in range(100, 400) if lab[s] is not None]; va = [s for s in range(400, 500) if lab[s] is not None]
allf = np.concatenate([F[s] for s in tr]); mu, sd = allf.mean(0), allf.std(0) + 1e-12
Ft = {s: torch.tensor((F[s] - mu) / sd, dtype=torch.float32) for s in S}
def fit(ss, tau, pen, epochs=300):
    th = torch.zeros(F[S[0]].shape[1], requires_grad=True); c = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam([th, c], lr=0.03)
    y = torch.tensor([lab[s] for s in ss]); Fs = torch.stack([Ft[s] for s in ss])
    for _ in range(epochs):
        sc = tau * torch.logsumexp(Fs @ th / tau, 1) - tau * np.log(128) + c
        loss = torch.nn.functional.binary_cross_entropy_with_logits(sc, y) + pen * (th ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return th.detach(), c.detach()
def score(th, c, ss, tau): return np.array([float(tau * torch.logsumexp(Ft[s] @ th / tau, 0) - tau * np.log(128) + c) for s in ss])
best = None; tr_arr = np.array(tr)
for tau in (0.1, 0.3, 1.0, 1000.0):
    for pen in (1e-3, 1e-2, 1e-1):
        aucs = []
        for f in range(5):
            te_m = np.arange(len(tr)) % 5 == f; a, b = list(tr_arr[~te_m]), list(tr_arr[te_m])
            th, c = fit(a, tau, pen); aucs.append(roc_auc_score([lab[s] for s in b], score(th, c, b, tau)))
        print('tau %g pen %g CV AUC %.3f' % (tau, pen, np.mean(aucs)), flush=True)
        if best is None or np.mean(aucs) > best[2]: best = (tau, pen, np.mean(aucs))
tau, pen, _ = best; th, c = fit(tr, tau, pen)
print('(1) selected tau %g pen %g: held-out AUC %.3f; on s 0-59 Spearman(score, T) %.2f' % (tau, pen, roc_auc_score([lab[s] for s in va], score(th, c, va, tau)), spearmanr(score(th, c, list(range(60)), tau), [T[s] for s in range(60)]).statistic))
print('(2) hand score: held-out AUC %.3f; on s 0-59 Spearman with T %.2f' % (roc_auc_score([lab[s] for s in va], [HAND[s] for s in va]), spearmanr([HAND[s] for s in range(60)], [T[s] for s in range(60)]).statistic))
names = [f'w_a{i}' for i in range(8)] + [f'w_b{i}' for i in range(8)] + ['bias'] + [f'w_a{i}^2' for i in range(8)] + [f'w_b{i}^2' for i in range(8)] + [f'w_a{i}*w_b{i}' for i in range(8)] + [f'|w_a{i}-w_b{i}|' for i in range(8)] + ['activity', 'act(1-act)']
o = np.argsort(-np.abs(th.numpy()))[:12]; print('largest unit-score coefficients:', ', '.join(f'{names[i]} {th[i]:+.2f}' for i in o))
np.save('c142_theta.npy', th.numpy())
