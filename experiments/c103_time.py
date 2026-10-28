# C-103b: C-98 with the compensated per-group Adam (lr x a, eps / a) for the rescaled net
# C-98: c34_wide.py plus an exact function-preserving rescaling (layer 1 x ALPHA, layer 2 weight / ALPHA) after the default init
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 100; BS = 256; STEPS = 10000; KS = [int(x) for x in os.environ.get("WIDTHS","384").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
class Net(nn.Module):
    def __init__(s, k):
        super().__init__(); s.k = k; d = 2 * n; L = []
        for _ in range(k): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        s.trunk = nn.Sequential(*L)
        def tower():
            M, dd = [], d
            for _ in range(3 - k): M += [nn.Linear(dd, 128), nn.ReLU()]; dd = 128
            return nn.Sequential(*M, nn.Linear(dd, 1))
        s.towers = nn.ModuleList([tower() for _ in range(n + 1)])
    def forward(s, x):
        h = s.trunk(x); return torch.cat([t(h) for t in s.towers], 1)
fp = f"{OUT}/wide_n8.csv"; done = set()
if os.path.exists(fp): done = {(r["k"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["k", "seed"] + [f"t{b}" for b in range(n + 1)])
for seed in range(5):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    for k in KS:
        if (str(k), str(seed)) in done: continue
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        net = nn.Sequential(nn.Linear(2*n,k),nn.ReLU(),nn.Linear(k,k),nn.ReLU(),nn.Linear(k,k),nn.ReLU(),nn.Linear(k,n+1)); ALPHA = float(os.environ.get("ALPHA","1"))
        with torch.no_grad(): net[0].weight.mul_(ALPHA); net[0].bias.mul_(ALPHA); net[2].weight.div_(ALPHA)
        LRB = float(os.environ.get("LR","1e-3")); EPS = 1e-8
        GROUPS = [dict(params=[net[0].weight, net[0].bias], lr=LRB*ALPHA, eps=EPS/ALPHA), dict(params=[net[2].weight], lr=LRB/ALPHA, eps=EPS*ALPHA),
                  dict(params=[net[2].bias, net[4].weight, net[4].bias, net[6].weight, net[6].bias], lr=LRB, eps=EPS)]
        opt = torch.optim.Adam(GROUPS)
        first = np.full(n + 1, -1); prev_ok = np.zeros(n + 1, bool); prev_t = 0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte) > 0).float() == Yte).float().mean(0).numpy()
                good = acc >= 0.99; newly = (first < 0) & prev_ok & good; first[newly] = prev_t; prev_ok = good; prev_t = step
                if (first >= 0).all(): break
        w.writerow([k, seed] + first.tolist()); fh.flush(); print(k, seed, first.tolist(), flush=True)
