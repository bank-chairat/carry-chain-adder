import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 100; BS = 256; STEPS = 4000; SEEDS = [int(x) for x in os.environ["SEEDS"].split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
P4 = bits(A ^ B, n)[:, [4]]; C4 = ((((A & 15) + (B & 15)) >> 4) & 1).astype(np.float32)[:, None]
fp = f"{OUT}/aux_n8.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["cond", "seed"] + [f"t{b}" for b in range(n + 1)])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    for cond, aux in (("p4", P4), ("c4", C4)):
        Y = np.concatenate([S, aux], 1)
        Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        L, d = [], 2 * n
        for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        net = nn.Sequential(*L, nn.Linear(d, Y.shape[1])); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        first = np.full(n + 1, -1); prev_ok = np.zeros(n + 1, bool); prev_t = 0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step == 1000: torch.save(net.state_dict(), f"{OUT}/net_{cond}_s{seed}_t1000.pt")
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte)[:, :n + 1] > 0).float() == Yte[:, :n + 1]).float().mean(0).numpy()
                good = acc >= 0.99; newly = (first < 0) & prev_ok & good; first[newly] = prev_t; prev_ok = good; prev_t = step
        w.writerow([cond, seed] + first.tolist()); fh.flush(); print(cond, seed, first.tolist(), flush=True)
