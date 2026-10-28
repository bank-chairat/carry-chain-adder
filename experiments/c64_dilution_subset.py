import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
TAG = os.environ.get("TAG", "all"); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; STEPS = int(os.environ.get("STEPS", "7000")); B2 = 0.999
LAMS = [float(x) for x in os.environ.get("LAMS", "1 0.3").split()]; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
fp = f"{OUT}/dilution_{TAG}_n8.csv"; done = set()
if os.path.exists(fp): done = {(r["lam"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["lam", "seed", "step", "acc4", "s_now", "I_cum", "t4_flag"])
oth = [int(x) for x in os.environ.get("OTH", "0 1 2 3 5 6 7 8").split()]
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
    for lam in LAMS:
        if (str(lam), str(seed)) in done: continue
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        layers = []; d = 2 * n
        for _ in range(3): layers.append(nn.Sequential(nn.Linear(d, 128), nn.ReLU())); d = 128
        head = nn.Linear(d, n + 1); trunk = [p for l in layers for p in l.parameters()]; opt = torch.optim.Adam(trunk + list(head.parameters()), lr=1e-3)
        m = torch.zeros(n + 1, dtype=torch.bool); m[torch.tensor(oth)] = True
        def fwd(x):
            for l in layers: x = l(x)
            hm = lam * x + (1 - lam) * x.detach()
            return torch.where(m, head(hm), head(x))
        v4 = [torch.zeros_like(p) for p in trunk]; vt = [torch.zeros_like(p) for p in trunk]
        prev_ok = False; prev_t = 0; I = 0.0; first = -1; acc4 = 0.0; s_now = 0.0; rows = []
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS); o = fwd(Xtr[i]); bce = nn.functional.binary_cross_entropy_with_logits(o, Ytr[i], reduction="none")
            l4 = bce[:, 4].mean(); lo = bce[:, oth].sum(1).mean()
            g4 = torch.autograd.grad(l4, trunk, retain_graph=True); go = torch.autograd.grad(lo, trunk, retain_graph=True)
            opt.zero_grad(); (l4 + lo).backward(); opt.step()
            with torch.no_grad():
                c = 1 - B2 ** step; num = 0.0; cnt = 0
                for k in range(len(trunk)):
                    v4[k].mul_(B2).add_((1 - B2) * g4[k] ** 2); vt[k].mul_(B2).add_((1 - B2) * (g4[k] + go[k]) ** 2)
                    r = torch.sqrt((v4[k] / c + 1e-16) / (vt[k] / c + 1e-16)); num += float(r.sum()); cnt += r.numel()
                s_now = num / cnt; I += s_now
            if step % EV == 0:
                with torch.no_grad(): acc4 = float(((fwd(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean())
                good = acc4 >= 0.99
                if prev_ok and good and first < 0: first = prev_t
                rows.append([lam, seed, step, round(acc4, 4), round(s_now, 4), round(I, 2), 0]); prev_ok = good; prev_t = step
                if first >= 0: break
        for r in rows: w.writerow(r)
        w.writerow([lam, seed, first, round(acc4, 4), round(s_now, 4), round(next((r[5] for r in rows if r[2] == first), -1), 2), 1]); fh.flush()
        print(lam, seed, "t4", first, "I(t4)", next((r[5] for r in rows if r[2] == first), None), flush=True)
