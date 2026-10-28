import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 100; BS = 256; STEPS = 10000; KS = [0, 3]; THR = [0.4, 0.2]; STEPS = 1500; EV = 25
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
fp = f"{OUT}/loss_n8.csv"; done = set()
if os.path.exists(fp): done = {(r["k"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["k", "seed", "thr"] + [f"t{b}" for b in range(n + 1)])
for seed in range(5):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    for k in KS:
        if (str(k), str(seed)) in done: continue
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        net = Net(k); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        firsts = {t: np.full(n + 1, -1) for t in THR}
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): bl = nn.functional.binary_cross_entropy_with_logits(net(Xte), Yte, reduction="none").mean(0).numpy()
                for t in THR:
                    nw = (firsts[t] < 0) & (bl < t); firsts[t][nw] = step
        for t in THR: w.writerow([k, seed, t] + firsts[t].tolist())
        fh.flush(); print(k, seed, {t: firsts[t].tolist() for t in THR}, flush=True)
