"""C-184: single-output nets (one net per sum bit, env BIT) with and without LayerNorm; otherwise as C-172.
Original C-172 docstring: per-bit learning times of the joint net with and without LayerNorm (does the 'middle bits last' hump flatten?).
C-116 recipe (n = 8, 3 x 128, Adam 1e-3, batch 256, split seed 500 + seed, init/batch seed seed*100+4), all 9 sum bits trained jointly,
held-out accuracy every 50 steps; bit time = first of two consecutive evaluations >= 0.99; stop when all bits are confirmed or at the cap.
Env NORM=none|ln, SEEDS, CAP. Usage: NORM=ln python3 c172_bits.py OUTDIR"""
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = int(os.environ.get("CAP", "10000")); NORM = os.environ.get("NORM", "none"); BIT = int(os.environ["BIT"])
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
fp = f"{OUT}/bits.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["norm", "bit", "seed", "t"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    NL = (lambda: [nn.LayerNorm(128)]) if NORM == "ln" else (lambda: [])
    net = nn.Sequential(nn.Linear(16, 128), *NL(), nn.ReLU(), nn.Linear(128, 128), *NL(), nn.ReLU(), nn.Linear(128, 128), *NL(), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); first = [None]; prev = [False]
    for step in range(1, CAP + 1):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i][:, BIT:BIT + 1], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xte) > 0).float() == Yte[:, BIT:BIT + 1]).float().mean(0).numpy()
            for b in range(1):
                good = acc[b] >= 0.99
                if first[b] is None and prev[b] and good: first[b] = step - EV
                prev[b] = good
            if all(f is not None for f in first): break
    t = first[0] if first[0] is not None else -1; w.writerow([NORM, BIT, seed, t]); fh.flush(); print(NORM, BIT, seed, t, flush=True)
