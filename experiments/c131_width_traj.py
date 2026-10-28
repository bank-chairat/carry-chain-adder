# C-131 (R3): joint runs for W128, DUP512 (4x duplicated width-128 init, ordinary Adam) and W512 (independent init), bit-4 logit-target correlation every 25 steps until bit 4 is confirmed.
import sys, os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 10000; K = 4
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; ARMS = os.environ.get("ARMS", "W128 DUP512 W512").split()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
def base(W): return nn.Sequential(nn.Linear(16, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, n + 1))
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te]); t4 = 2 * Y[te][:, 4] - 1
    for arm in ARMS:
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        if arm == "W128": net = base(128)
        elif arm == "W512": net = base(512)
        else:
            small = base(128); net = base(512)
            with torch.no_grad():
                net[0].weight.copy_(small[0].weight.repeat(K, 1)); net[0].bias.copy_(small[0].bias.repeat(K))
                for i in (2, 4): net[i].weight.copy_(small[i].weight.repeat(K, K) / K); net[i].bias.copy_(small[i].bias.repeat(K))
                net[6].weight.copy_(small[6].weight.repeat(1, K) / K); net[6].bias.copy_(small[6].bias)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); steps, corr = [], []; first = None; prev_ok = False; prev_t = 0
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
        np.savez(f"{OUT}/traj_{arm}_s{seed}.npz", steps=np.array(steps), corr=np.array(corr), t4=first if first is not None else -1); print(arm, seed, first, flush=True)
