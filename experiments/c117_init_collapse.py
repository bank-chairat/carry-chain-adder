# C-117 (W3): for the 40 initial weights of C-83 (same split, batch stream and init seeds), re-run the first 500 joint steps and record collapse measures; compare with the known C-83 bit-4 times.
import csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8; BS = 256
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); SUMV = (A + B).astype(np.float64)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, ytr = torch.tensor(X[tr]), torch.tensor(Y[tr]); Xd = Xtr[:8192]; Sd = SUMV[tr][:8192]
def r2_on(h, s):
    hc = h - h.mean(0); sc = (s - s.mean()) / s.std(); beta = (hc * sc[:, None]).mean(0); return float((beta ** 2).sum() / hc.var(0).sum())
def pr(h):
    hc = h - h.mean(0); lam = np.linalg.svd(hc, compute_uv=False) ** 2; return float(lam.sum() ** 2 / (lam ** 2).sum())
T = {}
for d in ("c83_a", "c83_b"):
    for r in csv.DictReader(open(f"{d}/init_align_joint_n8.csv")): T[int(r["s"])] = int(r["time"])
rows = []
for s in range(40):
    torch.manual_seed(10000 + s); rng = np.random.default_rng(777)
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); W20 = net[2].weight.detach().norm().item(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rec = {}
    for step in range(0, 501):
        if step > 0:
            i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step in (0, 100, 250, 500):
            with torch.no_grad(): h2 = net[:4](Xd).numpy().astype(np.float64); h3 = net[:6](Xd).numpy().astype(np.float64)
            rec[f"L2pr_{step}"] = pr(h2); rec[f"L2R2S_{step}"] = r2_on(h2, Sd); rec[f"L3R2S_{step}"] = r2_on(h3, Sd); rec[f"L2norm_{step}"] = net[2].weight.detach().norm().item() / W20
    rows.append(dict(s=s, time=T[s], **{k: round(v, 4) for k, v in rec.items()}))
with open("c117_init_collapse.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from scipy.stats import spearmanr
t = [r["time"] for r in rows]
for k in rows[0]:
    if k in ("s", "time"): continue
    print(f"{k:12s} Spearman with joint bit-4 time {spearmanr([r[k] for r in rows], t).statistic:+.2f}   range {min(r[k] for r in rows):.3f}-{max(r[k] for r in rows):.3f}")
