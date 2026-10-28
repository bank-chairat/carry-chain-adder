import sys, os, csv, json, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8; BS = 256; EV = 50; CAP = 9000; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
JOBS = [x.split(":") for x in os.environ["JOBS"].split()]       # s:arm
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); Xall = torch.tensor(X); F4 = torch.tensor(2 * Y[:, 4] - 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
def make(s):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
def unit_corrs(net):
    with torch.no_grad(): H = net[:4](Xall)
    Hc = H - H.mean(0); f = F4 - F4.mean(); return ((Hc * f[:, None]).sum(0) / (torch.sqrt((Hc ** 2).sum(0) * (f ** 2).sum()) + 1e-12)).numpy()
fp = f"{OUT}/seed.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["s", "arm", "unit", "corr_before", "corr_after", "change_norm", "t4"])
for s, arm in JOBS:
    s = int(s); net = make(s); c = unit_corrs(net); ac = np.abs(c); u = int(np.argmax(ac)); info = (float(c[u]), float("nan"), 0.0)
    g = np.random.default_rng(424242 + s); bound = 1 / np.sqrt(128)
    with torch.no_grad(): w_old = net[2].weight[u].clone(); b_old = net[2].bias[u].clone()
    if arm in ("remove_seed", "remove_median"):
        if arm == "remove_median": u = int(np.argsort(ac)[64])
        with torch.no_grad(): net[2].weight[u] = torch.tensor(g.uniform(-bound, bound, 128), dtype=torch.float32); net[2].bias[u] = float(g.uniform(-bound, bound))
    elif arm in ("plant_seed", "random_nudge"):
        h1 = net[:2](Xall).detach(); wv = net[2].weight[u].detach().clone().requires_grad_(True); bv = net[2].bias[u].detach().clone().reshape(1).requires_grad_(True)
        sign = 1.0 if c[u] >= 0 else -1.0; opt = torch.optim.SGD([wv, bv], lr=1e-3)
        for it in range(5000):
            a = torch.relu(h1 @ wv + bv); ac_ = a - a.mean(); f = F4 - F4.mean(); cc = (ac_ * f).sum() / (torch.sqrt((ac_ ** 2).sum() * (f ** 2).sum()) + 1e-12)
            if float(sign * cc) >= 0.065: break
            opt.zero_grad(); (-sign * cc).backward(); opt.step()
        dw = (wv.detach() - w_old); db = (bv.detach()[0] - b_old); norm = float(torch.sqrt((dw ** 2).sum() + db ** 2))
        if arm == "random_nudge":
            r = torch.tensor(g.normal(0, 1, 129), dtype=torch.float32); r = r / r.norm() * norm; dw, db = r[:128], r[128]
        with torch.no_grad(): net[2].weight[u] = w_old + dw; net[2].bias[u] = b_old + db
        info = (float(c[u]), float(unit_corrs(net)[u]), norm)
    if arm.startswith("remove"): info = (float(c[u]), float(unit_corrs(net)[u]), float(torch.sqrt(((net[2].weight[u] - w_old) ** 2).sum() + (net[2].bias[u] - b_old) ** 2)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777); first = None; prev_ok = False; prev_t = 0
    for step in range(1, CAP + 1):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xte)[:, 4] > 0).float() == yte[:, 4]).float().mean().item()
            good = acc >= 0.99
            if prev_ok and good: first = prev_t; break
            prev_ok = good; prev_t = step
    w.writerow([s, arm, u, round(info[0], 4), round(info[1], 4), round(info[2], 4), first if first is not None else -1]); fh.flush(); print(s, arm, u, [round(x, 4) for x in info], first, flush=True)
