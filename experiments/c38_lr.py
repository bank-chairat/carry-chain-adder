import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 100; BS = 256; STEPS = 10000; T = 4; WS = ['uniform', 'adaptive']
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
fp = f"{OUT}/lr_n8.csv"; done = set()
if os.path.exists(fp): done = {(r["w"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["w", "seed"] + [f"t{b}" for b in range(n + 1)])
for seed in range(10):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    for wt in WS:
        if (str(wt), str(seed)) in done: continue
        torch.manual_seed(seed * 100 + T); rng = np.random.default_rng(seed * 100 + T)
        L, d = [], 2 * n
        for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=float(os.environ.get('LR','1e-3')))
        wv = torch.ones(n + 1); ema = torch.full((n + 1,), 0.69)
        first = np.full(n + 1, -1); prev_ok = np.zeros(n + 1, bool); prev_t = 0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            per = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none"); loss = (per * wv).sum(1).mean(); bl = per.detach().mean(0); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                ema = 0.9 * ema + 0.1 * bl; wv = torch.clamp(ema / ema.mean(), 0.25, 8.0)
                if wt == 'shuffled': wv = wv[torch.tensor(rng.permutation(n + 1))]
                if wt == 'uniform': wv = torch.ones(n + 1)
                with torch.no_grad(): acc = ((net(Xte) > 0).float() == Yte).float().mean(0).numpy()
                good = acc >= 0.99; newly = (first < 0) & prev_ok & good; first[newly] = prev_t; prev_ok = good; prev_t = step
                if (first >= 0).all(): break
        w.writerow([wt, seed] + first.tolist()); fh.flush(); print(wt, seed, first.tolist(), flush=True)
