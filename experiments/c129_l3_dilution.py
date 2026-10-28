# C-129: ordinary Adam (own implementation, per-parameter choice of which gradient feeds the first and the second moment).
# arms: joint (both moments from the full gradient), blockL3 (layer 3: both moments from bit 4's gradient only), blockL3_dilV (layer 3: first moment from bit 4 only, second moment from the full gradient, i.e. as diluted as in the joint net).
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; STEPS = 6000; LR = 1e-3; B1, B2, EPS = 0.9, 0.999, 1e-8
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; ARMS = os.environ.get("ARMS", "joint blockL3 blockL3_dilV").split()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1); oth = [k for k in range(n + 1) if k != 4]
fp = f"{OUT}/l3dil.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["seed", "arm", "t4"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
    for arm in ARMS:
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        layers = []; d = 2 * n
        for _ in range(3): layers.append(nn.Sequential(nn.Linear(d, 128), nn.ReLU())); d = 128
        head = nn.Linear(d, n + 1); P = [p for l in layers for p in l.parameters()] + list(head.parameters()); L3 = set(id(p) for p in layers[2].parameters())
        m = [torch.zeros_like(p) for p in P]; v = [torch.zeros_like(p) for p in P]
        def fwd(x):
            for l in layers: x = l(x)
            return head(x)
        first = None; prev_ok = False; prev_t = 0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS); bce = nn.functional.binary_cross_entropy_with_logits(fwd(Xtr[i]), Ytr[i], reduction="none").mean(0)
            g4 = torch.autograd.grad(bce[4], P, retain_graph=True); go = torch.autograd.grad(bce[oth].sum(), P)
            c1 = 1 - B1 ** step; c2 = 1 - B2 ** step
            with torch.no_grad():
                for j, p in enumerate(P):
                    full = g4[j] + go[j]
                    if id(p) in L3 and arm == "blockL3": gm, gv = g4[j], g4[j]
                    elif id(p) in L3 and arm == "blockL3_dilV": gm, gv = g4[j], full
                    else: gm, gv = full, full
                    m[j].mul_(B1).add_((1 - B1) * gm); v[j].mul_(B2).add_((1 - B2) * gv * gv)
                    p.sub_(LR * (m[j] / c1) / (torch.sqrt(v[j] / c2) + EPS))
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([seed, arm, first if first is not None else -1]); fh.flush(); print(seed, arm, first, flush=True)
