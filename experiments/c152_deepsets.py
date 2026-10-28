# C-152 analysis: DeepSets model of escape (J500 >= 0.2) from layer-1 units (plus each unit's outgoing weight norm), trained on s 100-799, early-stopped on 800-899, tested on 900-1039 and on s 0-59 (known T).
import csv, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); torch.set_num_threads(2)
T = {}
for d in ('c83_a', 'c83_b'):
    for r in csv.DictReader(open(f'{d}/init_align_joint_n8.csv')): T[int(r['s'])] = int(r['time'])
for s in range(40, 60): T[s] = int(np.load(f'c127/traj_s{s}.npz')['t4'])
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float64)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel(); X = np.concatenate([bits(A, 8), bits(B, 8)], 1)
def w2norm(s):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, 9)); return net[2].weight.detach().numpy().astype(np.float64)
S = list(range(60)) + list(range(100, 1040)); F = {}; J = {}
for s in S:
    z = np.load(f'c139/init_s{s}.npz'); W = z['W1'].astype(np.float64); b = z['b1'].astype(np.float64); act = ((X @ W.T + b) > 0).mean(0); W2 = w2norm(s)
    F[s] = np.c_[W, b, act, np.linalg.norm(W2, axis=0)]; J[s] = float(z['J'][-1])
lab = {s: (1.0 if J[s] >= 0.2 else (0.0 if J[s] < 0.05 else None)) for s in S}
tr = [s for s in range(100, 800) if lab[s] is not None]; va = [s for s in range(800, 900) if lab[s] is not None]; te = [s for s in range(900, 1040) if lab[s] is not None]
allf = np.concatenate([F[s] for s in tr]); mu, sd = allf.mean(0), allf.std(0) + 1e-9
Ft = {s: torch.tensor((F[s] - mu) / sd, dtype=torch.float32) for s in S}
class DS(nn.Module):
    def __init__(s, d):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()); s.rho = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x):  # x: (batch, units, d)
        h = s.phi(x); return s.rho(torch.cat([h.mean(1), h.max(1).values], 1))[:, 0]
def batch(ss): return torch.stack([Ft[s] for s in ss])
best = (-1, None); model = DS(Ft[S[0]].shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
ytr = torch.tensor([lab[s] for s in tr]); Xtr = batch(tr); Xva = batch(va); yva = [lab[s] for s in va]
for ep in range(300):
    perm = torch.randperm(len(tr))
    for k in range(0, len(tr), 32):
        idx = perm[k:k + 32]; loss = nn.functional.binary_cross_entropy_with_logits(model(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): auc = roc_auc_score(yva, model(Xva).numpy())
    if auc > best[0]: best = (auc, {k: v.clone() for k, v in model.state_dict().items()}, ep)
model.load_state_dict(best[1])
with torch.no_grad(): ste = model(batch(te)).numpy(); sold = model(batch(list(range(60)))).numpy(); str_ = model(Xtr).numpy()
print('train n %d (escaped %d), val n %d, test n %d' % (len(tr), int(sum(lab[s] for s in tr)), len(va), len(te)))
print('best validation AUC %.3f at epoch %d; train AUC %.3f' % (best[0], best[2], roc_auc_score(ytr.numpy(), str_)))
print('(1) TEST AUC %.3f; (2) Spearman(score, T) on s 0-59: %.2f' % (roc_auc_score([lab[s] for s in te], ste), spearmanr(sold, [T[s] for s in range(60)]).statistic))
