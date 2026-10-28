# C-154 (width, new-units half): interpolate between the 4x duplicated width-128 initialisation (no new features) and an independent width-512 initialisation:
# theta(alpha) = sqrt(1 - alpha^2) * theta_dup + alpha * theta_iid, every parameter; joint training (Adam 1e-3, batch 256); bit-4 logit-target correlation every 25 steps until bit 4 is confirmed.
import sys, os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 10000; K = 4
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0.1 0.3 0.5 0.7").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
def base(W): return nn.Sequential(nn.Linear(16, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, n + 1))
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te]); t4 = 2 * Y[te][:, 4] - 1
    for al in ALPHAS:
        torch.manual_seed(seed * 100 + 4); small = base(128); dup = base(512)
        with torch.no_grad():
            dup[0].weight.copy_(small[0].weight.repeat(K, 1)); dup[0].bias.copy_(small[0].bias.repeat(K))
            for i in (2, 4): dup[i].weight.copy_(small[i].weight.repeat(K, K) / K); dup[i].bias.copy_(small[i].bias.repeat(K))
            dup[6].weight.copy_(small[6].weight.repeat(1, K) / K); dup[6].bias.copy_(small[6].bias)
        torch.manual_seed(777000 + seed); iid = base(512); net = base(512)
        with torch.no_grad():
            for p, a, b in zip(net.parameters(), dup.parameters(), iid.parameters()): p.copy_(np.sqrt(1 - al ** 2) * a + al * b)
        rng = np.random.default_rng(seed * 100 + 4); opt = torch.optim.Adam(net.parameters(), lr=1e-3); steps, corr = [], []; first = None; prev_ok = False; prev_t = 0
        for step in range(0, CAP + 1):
            if step > 0:
                i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0:
                with torch.no_grad(): z = net(Xte)[:, 4].numpy().astype(np.float64)
                steps.append(step); corr.append(float(np.corrcoef(z, t4)[0, 1]) if z.std() > 0 else 0.0)
                if step % EV == 0 and step > 0:
                    good = float(((z > 0) == (t4 > 0)).mean()) >= 0.99
                    if prev_ok and good and first is None: first = prev_t
                    prev_ok = good; prev_t = step
                    if first is not None: break
        np.savez(f"{OUT}/traj_a{al}_s{seed}.npz", steps=np.array(steps), corr=np.array(corr), t4=first if first is not None else -1); print(al, seed, first, flush=True)
