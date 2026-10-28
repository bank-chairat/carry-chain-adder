import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8; BS = 256; EV = 50
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); MODE = os.environ["MODE"]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); oth = [k for k in range(9) if k != 4]
jobs = []
if MODE == "inits":
    for s in range(40): jobs.append(("C83", s, 128, 1e-3, 9000))
else:
    for name, W, LR, CAP in (("lr3e-4", 128, 3e-4, 10000), ("lr1e-2", 128, 1e-2, 3000), ("W64", 64, 1e-3, 10000), ("W512", 512, 1e-3, 3000)):
        for seed in range(5): jobs.append((name, seed, W, LR, CAP))
for name, s, W, LR, CAP in jobs:
    if name == "C83":
        perm = np.random.default_rng(500).permutation(len(A)); torch.manual_seed(10000 + s); rng = np.random.default_rng(777)
    else:
        perm = np.random.default_rng(500 + s).permutation(len(A)); torch.manual_seed(s * 100 + 4); rng = np.random.default_rng(s * 100 + 4)
    tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]); t4 = 2 * Y[te][:, 4] - 1; Xp, Yp = Xtr[:8192], Ytr[:8192]
    L1 = nn.Sequential(nn.Linear(16, W), nn.ReLU()); L2 = nn.Sequential(nn.Linear(W, W), nn.ReLU()); L3 = nn.Sequential(nn.Linear(W, W), nn.ReLU()); head = nn.Linear(W, 9)
    net = nn.Sequential(L1, L2, L3, head); body = [p for m in (L1, L2, L3) for p in m.parameters()]; opt = torch.optim.Adam(net.parameters(), lr=LR)
    steps, P, Jc = [], [], []; first = None; prev_ok = False; prev_t = 0
    for step in range(0, CAP + 1):
        if step > 0:
            i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            lo = nn.functional.binary_cross_entropy_with_logits(net(Xp), Yp, reduction="none")[:, oth].sum(1).mean(); g = torch.autograd.grad(lo, body)
            P.append(float(torch.sqrt(sum((x ** 2).sum() for x in g))))
            with torch.no_grad(): z = net(Xte)[:, 4].numpy().astype(np.float64)
            Jc.append(float(np.corrcoef(z, t4)[0, 1]) if z.std() > 0 else 0.0); steps.append(step)
            if step % EV == 0 and step > 0:
                good = float(((z > 0) == (t4 > 0)).mean()) >= 0.99
                if prev_ok and good and first is None: first = prev_t
                prev_ok = good; prev_t = step
                if first is not None: break
    np.savez(f"{OUT}/push_{name}_s{s}.npz", steps=np.array(steps), P=np.array(P), J=np.array(Jc), t4=first if first is not None else -1); print(name, s, first, flush=True)
