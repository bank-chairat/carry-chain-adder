import csv, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2)
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = torch.tensor(np.concatenate([bits(A, 8), bits(B, 8)], 1)); F4 = (2 * (((A + B) >> 4) & 1) - 1).astype(np.float64)
T = {}
for d in ('c83_a', 'c83_b'):
    for r in csv.DictReader(open(f'{d}/init_align_joint_n8.csv')): T[int(r['s'])] = int(r['time'])
for s in range(40, 60): T[s] = int(np.load(f'c127/traj_s{s}.npz')['t4'])
used = {28, 26, 38, 8, 11, 22, 35, 5, 37, 36, 10, 25, 24}
def stats(s, layer_end=4):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, 9))
    with torch.no_grad(): H = net[:layer_end](X).numpy().astype(np.float64)
    Hc = H - H.mean(0); f = F4 - F4.mean(); c = (Hc * f[:, None]).sum(0) / (np.sqrt((Hc ** 2).sum(0) * (f ** 2).sum()) + 1e-12)
    return c[np.argmax(np.abs(c))], c.mean()
new = [s for s in range(100, 500)]; old = [s for s in range(60) if s not in used]
J = {s: float(np.load(f'c139/init_s{s}.npz')['J'][-1]) for s in new + old}
for le, name in ((4, 'layer 2'), (2, 'layer 1'), (6, 'layer 3')):
    st = {s: stats(s, le) for s in new + old}
    for k, nm in ((0, 'sgn1'), (1, 'tilt')):
        r1 = spearmanr([st[s][k] for s in new], [J[s] for s in new]).statistic; r2 = spearmanr([st[s][k] for s in old], [T[s] for s in old]).statistic
        lab = [(s, 1 if J[s] >= 0.2 else 0) for s in new if J[s] >= 0.2 or J[s] < 0.05]; auc = roc_auc_score([l for _, l in lab], [st[s][k] for s, _ in lab])
        print(f'{name} {nm}: Spearman with J500 (s 100-499) {r1:+.2f}; escape AUC {auc:.2f}; Spearman with T (s 0-59 minus used, n={len(old)}) {r2:+.2f}', flush=True)
