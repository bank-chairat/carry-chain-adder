import csv, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
torch.set_num_threads(1); n = 8; BS = 256
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
T = {}
for d in ("c83_a", "c83_b"):
    for r in csv.DictReader(open(f"{d}/init_align_joint_n8.csv")): T[int(r["s"])] = int(r["time"])
def make(s):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
rows = []
for s in range(40):
    rec = {}
    for mode, steps in (("joint", 500), ("alone", 200)):
        net = make(s); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777)
        for step in range(0, steps + 1):
            if step > 0:
                i = rng.integers(0, len(tr), BS); z = net(Xtr[i])
                loss = nn.functional.binary_cross_entropy_with_logits(z, ytr[i], reduction="none").sum(1).mean() if mode == "joint" else nn.functional.binary_cross_entropy_with_logits(z[:, 4], ytr[i][:, 4])
                opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0:
                with torch.no_grad(): z4 = net(Xte)[:, 4]; rec[f"{mode}_bce_{step}"] = float(nn.functional.binary_cross_entropy_with_logits(z4, yte[:, 4])); rec[f"{mode}_acc_{step}"] = float(((z4 > 0).float() == yte[:, 4]).float().mean())
    rows.append(dict(s=s, time=T[s], **{k: round(v, 5) for k, v in rec.items()}))
with open("c123_race.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
t = [r["time"] for r in rows]
for mode, steps in (("joint", 500), ("alone", 200)):
    print(mode, "Spearman of held-out bit-4 BCE with joint time:", " ".join(f"{st}:{spearmanr([r[f'{mode}_bce_{st}'] for r in rows], t).statistic:+.2f}" for st in range(25, steps + 1, 25)))
    print(mode, "median bit-4 acc:", " ".join(f"{st}:{np.median([r[f'{mode}_acc_{st}'] for r in rows]):.3f}" for st in range(25, steps + 1, 25)))
