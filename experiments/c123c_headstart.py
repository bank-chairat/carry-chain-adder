# C-123c (O2, causal): does a small head start for bit 4 make a slow initialisation fast? K steps on one bit only (bit 4, or bit 3 as control), then ordinary joint training with the same optimiser and batch stream (C-83 recipe).
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 9000; K = int(os.environ.get("K", "20"))
INITS = [int(x) for x in os.environ["INITS"].split()]; ARMS = os.environ.get("ARMS", "pre4 pre3").split()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te]); t4 = Y[te][:, 4]
fp = f"{OUT}/headstart.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["s", "arm", "K", "t4", "acc4_at_switch", "corr4_at_switch"])
for s in INITS:
    for arm in ARMS:
        torch.manual_seed(10000 + s); L, d = [], 16
        for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777)
        tgt = {"pre4": 4, "pre3": 3}[arm]
        for step in range(1, K + 1):
            i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i])[:, tgt], ytr[i][:, tgt]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): z = net(Xte)[:, 4].numpy().astype(np.float64)
        acc0 = float(((z > 0) == (t4 > 0.5)).mean()); corr0 = float(np.corrcoef(z, 2 * t4 - 1)[0, 1])
        first = None; prev_ok = False; prev_t = 0
        for step in range(K + 1, CAP + 1):
            i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte)[:, 4] > 0).float() == yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([s, arm, K, first if first is not None else -1, round(acc0, 4), round(corr0, 4)]); fh.flush(); print(s, arm, K, first, round(acc0, 3), round(corr0, 3), flush=True)
