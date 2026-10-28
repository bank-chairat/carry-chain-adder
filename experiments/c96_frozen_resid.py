import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; STEPS = int(os.environ.get("STEPS", "6000")); SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel(); X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
fp = f"{OUT}/frozen_n8.csv"; done = set()
if os.path.exists(fp): done = {r["seed"] for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["seed"] + [f"t{b}" for b in range(n + 1)] + ["final_acc4"])
oth = torch.tensor([b for b in range(n + 1) if b != 4])
for seed in SEEDS:
    if str(seed) in done: continue
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with torch.no_grad(): R0 = torch.sigmoid(net(Xtr)) - Ytr          # initial residual of every output on every training input
    first = np.full(n + 1, -1); prev_ok = np.zeros(n + 1, bool); prev_t = 0; acc = None
    for step in range(1, STEPS + 1):
        i = rng.integers(0, len(tr), BS); z = net(Xtr[i])
        loss = nn.functional.binary_cross_entropy_with_logits(z[:, 4], Ytr[i][:, 4]) + (R0[i][:, oth] * z[:, oth]).mean(0).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xte) > 0).float() == Yte).float().mean(0).numpy()
            good = acc >= 0.99; newly = (first < 0) & prev_ok & good; first[newly] = prev_t; prev_ok = good; prev_t = step
            if first[4] >= 0: break
    w.writerow([seed] + first.tolist() + [round(float(acc[4]), 4)]); fh.flush(); print(seed, first.tolist(), "bit4 acc %.4f" % acc[4], flush=True)
