# C-123b: same runs as C-123 (joint only), with scale-free measures of bit-4 progress: Pearson correlation of the bit-4 logit with the +-1 target and the AUC, plus the logit standard deviation (held-out set).
import csv, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr, rankdata
torch.set_num_threads(1); n = 8; BS = 256
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te]); t4 = Y[te][:, 4]
T = {}
for d in ("c83_a", "c83_b"):
    for r in csv.DictReader(open(f"{d}/init_align_joint_n8.csv")): T[int(r["s"])] = int(r["time"])
def auc(z, y):
    r = rankdata(z); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
rows = []
for s in range(40):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777); rec = {}
    for step in range(0, 501):
        if step > 0:
            i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            with torch.no_grad(): z = net(Xte)[:, 4].numpy().astype(np.float64)
            rec[f"corr_{step}"] = float(np.corrcoef(z, 2 * t4 - 1)[0, 1]) if z.std() > 0 else 0.0; rec[f"auc_{step}"] = auc(z, t4); rec[f"zstd_{step}"] = float(z.std())
    rows.append(dict(s=s, time=T[s], **{k: round(v, 5) for k, v in rec.items()}))
with open("c123b_race.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
t = [r["time"] for r in rows]
for k in ("corr", "auc", "zstd"):
    print(k, "Spearman with joint time:", " ".join(f"{st}:{spearmanr([r[f'{k}_{st}'] for r in rows], t).statistic:+.2f}" for st in range(25, 501, 25)))
    print(k, "median:", " ".join(f"{st}:{np.median([r[f'{k}_{st}'] for r in rows]):.3f}" for st in range(25, 501, 50)))
