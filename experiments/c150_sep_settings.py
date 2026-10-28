# C-150: c114_sep_block.py with learning-rate and frozen-layer options (to measure the movement cost M(r) in other settings)
# C-114: C-73 (separate Adam moments per output in the shared layers, full step per output) with the other outputs' updates removed from chosen hidden layers; bit 4's own update is always applied everywhere.
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; STEPS = int(os.environ.get('STEPS', '8000')); LR = float(os.environ.get('LR', '1e-3')); FW = int(os.environ.get('FW', '0')); B1, B2, EPS = 0.9, 0.999, 1e-8
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; ARMS = os.environ.get("ARMS", "none L1 L2 L3 L23 all").split(); FO = float(os.environ.get("FO", "1"))
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
fp = f"{OUT}/sepblock.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["seed", "blocked_layers", "t4", "FO", "LR", "FW"])
LAYER_OF = [0, 0, 1, 1, 2, 2]                      # trunk parameter j -> hidden layer index (W, b per layer)
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
    for arm in ARMS:
        blocked = {"none": set(), "L1": {0}, "L2": {1}, "L3": {2}, "L23": {1, 2}, "L12": {0, 1}, "all": {0, 1, 2}}[arm]
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        if FW:
            frozen = nn.Sequential(nn.Linear(2 * n, FW), nn.ReLU())
            for p in frozen.parameters(): p.requires_grad_(False)
            d = FW
        else: frozen = nn.Identity(); d = 2 * n
        layers = []
        for _ in range(3): layers.append(nn.Sequential(nn.Linear(d, 128), nn.ReLU())); d = 128
        head = nn.Linear(d, n + 1); trunk = [p for l in layers for p in l.parameters()]; hp = list(head.parameters())
        opt_h = torch.optim.Adam(hp, lr=LR); K = n + 1
        m = [[torch.zeros_like(p) for p in trunk] for _ in range(K)]; v = [[torch.zeros_like(p) for p in trunk] for _ in range(K)]
        def fwd(x):
            x = frozen(x)
            for l in layers: x = l(x)
            return head(x)
        first = None; prev_ok = False; prev_t = 0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS); o = fwd(Xtr[i]); bce = nn.functional.binary_cross_entropy_with_logits(o, Ytr[i], reduction="none").mean(0)
            upd = [torch.zeros_like(p) for p in trunk]; hg = [torch.zeros_like(p) for p in hp]; c1 = 1 - B1 ** step; c2 = 1 - B2 ** step
            for k in range(K):
                g = torch.autograd.grad(bce[k], trunk + hp, retain_graph=(k < K - 1))
                for j in range(len(trunk)):
                    gj = g[j]; m[k][j].mul_(B1).add_((1 - B1) * gj); v[k][j].mul_(B2).add_((1 - B2) * gj * gj)
                    if k != 4 and LAYER_OF[j] in blocked: continue
                    upd[j] += (1.0 if k == 4 else FO) * (m[k][j] / c1) / (torch.sqrt(v[k][j] / c2) + EPS)
                for j in range(len(hp)): hg[j] += g[len(trunk) + j]
            with torch.no_grad():
                for j, p in enumerate(trunk): p.sub_(LR * upd[j])
            for p, gg in zip(hp, hg): p.grad = gg
            opt_h.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([seed, arm, first if first is not None else -1, FO, LR, FW]); fh.flush(); print(seed, arm, first, flush=True)
