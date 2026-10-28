import json, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
import csv
torch.set_num_threads(2); n = 8; BS = 256
on = {r['s']: r for r in json.load(open('c125_onsets.json'))}; pick = sorted(on, key=lambda s: on[s]['t_on'])[:6]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); F4 = 2 * Y[:, 4] - 1
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]); t4 = 2 * Y[te][:, 4] - 1; Xall = torch.tensor(X)
def make(s):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
def unit_corr(H):
    Hc = H - H.mean(0); return (Hc * (F4 - F4.mean())[:, None]).sum(0) / (np.sqrt((Hc ** 2).sum(0) * ((F4 - F4.mean()) ** 2).sum()) + 1e-12)
def corr(z): return float(np.corrcoef(z.numpy().astype(np.float64), t4)[0, 1])
print('(a) traced runs')
for s in pick:
    net = make(s)
    with torch.no_grad(): c0 = np.abs(unit_corr(net[:4](Xall).numpy().astype(np.float64)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777)
    for step in range(1, on[s]['t_on'] + 101):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Jb = corr(net(Xte)[:, 4]); mean = net[:4](Xtr).mean(0); drops = np.zeros(128)
        for u in range(128):
            h = net[:4](Xte).clone(); h[:, u] = mean[u]; drops[u] = Jb - corr(net[4:](h)[:, 4])
    top = np.argsort(-drops)[:5]; pc = [float((c0 < c0[u]).mean() * 100) for u in top]
    print(f'  s {s}: top-5 layer-2 units, initial |corr with bit-4 target| percentiles {[round(x) for x in pc]}, median {np.median(pc):.0f}; their initial |corr| {np.round(c0[top], 4).tolist()} (layer max {c0.max():.4f})', flush=True)
T = {}
for d in ('c83_a', 'c83_b'):
    for r in csv.DictReader(open(f'{d}/init_align_joint_n8.csv')): T[int(r['s'])] = int(r['time'])
for s in range(40, 60): T[s] = int(np.load(f'c127/traj_s{s}.npz')['t4'])
G = {1: {}, 2: {}, 3: {}}; Jv = {}
for s in list(range(60)) + list(range(100, 500)):
    net = make(s)
    with torch.no_grad():
        for li, idx in ((1, 2), (2, 4), (3, 6)): c = np.sort(np.abs(unit_corr(net[:idx](Xall).numpy().astype(np.float64))))[::-1]; G[li][s] = c[:5].mean()
    Jv[s] = float(np.load(f'c139/init_s{s}.npz')['J'][-1])
new = list(range(100, 500)); old = list(range(60))
for li in (1, 2, 3):
    print('(b) layer %d: Spearman(G, J500) over s 100-499 %+.2f; Spearman(G, T) over s 0-59 %+.2f; G median %.4f' % (li, spearmanr([G[li][s] for s in new], [Jv[s] for s in new]).statistic, spearmanr([G[li][s] for s in old], [T[s] for s in old]).statistic, np.median([G[li][s] for s in new])))
