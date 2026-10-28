# C-125 (O1/O2) [env NORM=ln adds a LayerNorm before each ReLU, C-182]: full joint runs for the 40 C-83 initialisations with the bit-4 logit-target correlation recorded every 25 steps until bit 4 is confirmed (C-83 recipe).
import sys, os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8; BS = 256; EV = 50; CAP = 9000
INITS = [int(x) for x in os.environ.get("INITS", " ".join(str(i) for i in range(40))).split()]; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te]); t4 = 2 * Y[te][:, 4] - 1
for s in INITS:
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), *([nn.LayerNorm(128)] if os.environ.get('NORM', 'none') == 'ln' else []), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777)
    steps, corr, acc4 = [], [], []; first = None; prev_ok = False; prev_t = 0
    for step in range(0, CAP + 1):
        if step > 0:
            i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            with torch.no_grad(): z = net(Xte)[:, 4].numpy().astype(np.float64)
            a = float(((z > 0) == (t4 > 0)).mean()); steps.append(step); corr.append(float(np.corrcoef(z, t4)[0, 1]) if z.std() > 0 else 0.0); acc4.append(a)
            if step % EV == 0 and step > 0:
                good = a >= 0.99
                if prev_ok and good and first is None: first = prev_t
                prev_ok = good; prev_t = step
                if first is not None: break
    np.savez(f"{OUT}/traj_s{s}.npz", steps=np.array(steps), corr=np.array(corr), acc4=np.array(acc4), t4=first if first is not None else -1)
    print(s, first, flush=True)
