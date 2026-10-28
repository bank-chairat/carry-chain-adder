"""C-181: per-initialisation S-dominance at step 250 (C-83 recipe: split seed 500, batch stream 777, torch seed 10000 + s) for s = 100-299, against the learning time T of the same runs (C-162)."""
import numpy as np, torch, torch.nn as nn, glob, re, csv, sys
from scipy.stats import spearmanr
torch.set_num_threads(1); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel(); X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); SV = (A + B).astype(float)
perm = np.random.default_rng(500).permutation(len(A)); tr = perm[:len(A) // 2]; Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr]); s8 = SV[tr][:8192]
def r2_on(h, s):
    hc = h - h.mean(0); sc = (s - s.mean()) / s.std(); beta = (hc * sc[:, None]).mean(0); return float((beta ** 2).sum() / hc.var(0).sum())
T = {int(re.findall(r"traj_s(\d+)", f)[0]): int(np.load(f)["t4"]) for f in glob.glob("c162_*/traj_s*.npz")}
out = csv.writer(open("c181_metric.csv", "w", newline="")); out.writerow(["s", "T", "L2_R2S250", "L3_R2S250", "L3norm_binR2_250"])
rows = []
for s in range(100, 300):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777)
    for step in range(1, 251):
        i = rng.integers(0, len(tr), 256); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): h2 = net[:4](Xtr[:8192]).numpy().astype(float); h3 = net[:6](Xtr[:8192]).numpy().astype(float)
    nrm = np.linalg.norm(h3, axis=1); b = np.minimum((s8 / 32).astype(int), 15); means = np.array([nrm[b == k].mean() for k in range(16)]); nb = 1 - ((nrm - means[b]) ** 2).mean() / nrm.var()
    r = [s, T[s], r2_on(h2, s8), r2_on(h3, s8), nb]; rows.append(r); out.writerow(r)
R = np.array(rows)
for k, name in [(2, "layer-2 R^2 on S"), (3, "layer-3 R^2 on S"), (4, "layer-3 norm explained by S bins")]:
    print(f"Spearman({name} at 250, T) over {len(R)} runs: {spearmanr(R[:, k], R[:, 1]).statistic:+.2f}  (range {R[:, k].min():.3f}-{R[:, k].max():.3f})")
