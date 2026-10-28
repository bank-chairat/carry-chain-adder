# C-175: per-bit learning times at n = 12 (C-149 data recipe), joint net, with and without LayerNorm (env NORM=none|ln). (alone, conf3, joint) at n = 12 with the C-48 data recipe (300,000 sampled training pairs, 20,000 held-out pairs, 13 outputs).
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 12; EV = 100; BS = 256; STEPS = 20000; ARMS = os.environ.get("ARMS", "alone conf3 joint").split(); SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
_r = np.random.default_rng(12345); _ids = np.unique(_r.integers(0, 1 << (2 * n), 330000))[:320000]; _r.shuffle(_ids)
A = _ids >> n; B = _ids & ((1 << n) - 1)
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
tr, te = np.arange(300000), np.arange(300000, 320000)
fp = f"{OUT}/bits12.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
NORM = os.environ.get("NORM", "none")
if new: w.writerow(["norm", "seed"] + [f"t{i}" for i in range(n + 1)])
Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
for seed in SEEDS:
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    NL = (lambda: [nn.LayerNorm(128)]) if NORM == "ln" else (lambda: [])
    net = nn.Sequential(nn.Linear(2 * n, 128), *NL(), nn.ReLU(), nn.Linear(128, 128), *NL(), nn.ReLU(), nn.Linear(128, 128), *NL(), nn.ReLU(), nn.Linear(128, n + 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); first = [None] * (n + 1); prev = [False] * (n + 1)
    for step in range(1, STEPS + 1):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xte) > 0).float() == Yte).float().mean(0).numpy()
            for b in range(n + 1):
                good = acc[b] >= 0.99
                if first[b] is None and prev[b] and good: first[b] = step - EV
                prev[b] = good
            if all(f is not None for f in first): break
    row = [f if f is not None else -1 for f in first]; w.writerow([NORM, seed] + row); fh.flush(); print(NORM, seed, row, flush=True)
