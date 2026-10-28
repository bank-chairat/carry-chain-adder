import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
W = int(os.environ.get("WIDTH", "128")); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; STEPS = int(os.environ.get("STEPS", "6000")); F = float(os.environ.get("F", "1"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; LR = 1e-3; B1, B2, EPS = 0.9, 0.999, 1e-8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
fp = f"{OUT}/sepmoments_n8.csv"; done = set()
if os.path.exists(fp): done = {r["seed"] for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["f", "seed"] + [f"t{b}" for b in range(n + 1)] + ["final_acc4"])
for seed in SEEDS:
    if str(seed) in done: continue
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    layers = []; d = 2 * n
    for _ in range(3): layers.append(nn.Sequential(nn.Linear(d, W), nn.ReLU())); d = W
    head = nn.Linear(d, n + 1); trunk = [p for l in layers for p in l.parameters()]; hp = list(head.parameters())
    opt_h = torch.optim.Adam(hp, lr=LR)
    K = n + 1; m = [[torch.zeros_like(p) for p in trunk] for _ in range(K)]; v = [[torch.zeros_like(p) for p in trunk] for _ in range(K)]
    def fwd(x):
        for l in layers: x = l(x)
        return head(x)
    first = np.full(n + 1, -1); prev_ok = np.zeros(n + 1, bool); prev_t = 0; acc = None
    for step in range(1, STEPS + 1):
        i = rng.integers(0, len(tr), BS); o = fwd(Xtr[i]); bce = nn.functional.binary_cross_entropy_with_logits(o, Ytr[i], reduction="none").mean(0)   # (K,) batch means
        upd = [torch.zeros_like(p) for p in trunk]; hg = [torch.zeros_like(p) for p in hp]
        c1 = 1 - B1 ** step; c2 = 1 - B2 ** step
        for k in range(K):
            g = torch.autograd.grad(bce[k], trunk + hp, retain_graph=(k < K - 1))
            for j in range(len(trunk)):
                gj = g[j]; m[k][j].mul_(B1).add_((1 - B1) * gj); v[k][j].mul_(B2).add_((1 - B2) * gj * gj)
                upd[j] += (m[k][j] / c1) / (torch.sqrt(v[k][j] / c2) + EPS)
            for j in range(len(hp)): hg[j] += g[len(trunk) + j]
        with torch.no_grad():
            for j, p in enumerate(trunk): p.sub_(LR * F * upd[j])
        for p, gg in zip(hp, hg): p.grad = gg
        opt_h.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((fwd(Xte) > 0).float() == Yte).float().mean(0).numpy()
            good = acc >= 0.99; newly = (first < 0) & prev_ok & good; first[newly] = prev_t; prev_ok = good; prev_t = step
            if (first >= 0).all(): break
    w.writerow([F, seed] + first.tolist() + [round(float(acc[4]), 4)]); fh.flush(); print(F, seed, first.tolist(), flush=True)
