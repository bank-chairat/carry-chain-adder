import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = int(os.environ.get("BS", "256")); K = float(os.environ.get("K", "1")); STEPS = int(os.environ.get("STEPS", "6000"))
LAMS = [float(x) for x in os.environ.get("LAMS", "0 0.03 0.1 0.3 1").split()]; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
fp = f"{OUT}/hiddenlr_k{K}_n8.csv"; done = set()
if os.path.exists(fp): done = {(r["lam"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["lam", "seed"] + [f"t{b}" for b in range(n + 1)] + ["final_acc4"])
others = torch.tensor([b for b in range(n + 1) if b != 4])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
    for lam in LAMS:
        if (str(lam), str(seed)) in done: continue
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        layers = []; d = 2 * n
        for _ in range(3): layers.append(nn.Sequential(nn.Linear(d, 128), nn.ReLU())); d = 128
        head = nn.Linear(d, n + 1); params = [p for l in layers for p in l.parameters()] + list(head.parameters()); opt = torch.optim.Adam([{"params": [p for l in layers for p in l.parameters()], "lr": 1e-3 * K}, {"params": list(head.parameters()), "lr": 1e-3}])
        def fwd(x):
            for l in layers: x = l(x)
            hm = lam * x + (1 - lam) * x.detach()          # same value as x; gradient into the trunk scaled by lam
            o_full = head(x); o_oth = head(hm)               # bit 4 uses x; all other outputs use hm
            m = torch.zeros(n + 1, dtype=torch.bool); m[others] = True
            return torch.where(m, o_oth, o_full)
        first = np.full(n + 1, -1); prev_ok = np.zeros(n + 1, bool); prev_t = 0; acc = None
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(Xte) > 0).float() == Yte).float().mean(0).numpy()
                good = acc >= 0.99; newly = (first < 0) & prev_ok & good; first[newly] = prev_t; prev_ok = good; prev_t = step
                if (first >= 0).all(): break
        w.writerow([lam, seed] + first.tolist() + [round(float(acc[4]), 4)]); fh.flush(); print(lam, seed, first.tolist(), flush=True)
