# C-100: held-out error of each sum bit i (1..7), stratified by active carry-chain length L_i, along training of the joint net (n=8, 3x128, Adam 1e-3, batch 256, seeds 0-4, same split/init as C-39).
# L_i = 0 if c_i = 0, else 1 + (number of consecutive propagate positions i-1, i-2, ... below bit i, stopping at the first non-propagate position).
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; STEPS = int(os.environ.get("STEPS", "5000"))
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
Pb = (A ^ B); G = A & B
def chainlen(i):
    L = np.zeros(len(A), int)
    for k in range(len(A)):
        a, b = int(A[k]), int(B[k]); j = i - 1
        while j >= 0 and ((a >> j) & 1) != ((b >> j) & 1): j -= 1
        L[k] = (i - j) if (j >= 0 and ((a >> j) & 1) and ((b >> j) & 1)) else 0     # generator at j, propagates j+1..i-1 -> length i-j
    return L
LS = {i: chainlen(i) for i in range(1, 8)}
fp = f"{OUT}/strata.csv"; done = set()
if os.path.exists(fp): done = {r["seed"] for r in csv.DictReader(open(fp))}
fh = open(fp, "a", newline=""); w = csv.writer(fh)
if not done: w.writerow(["seed", "step", "bit", "L", "count", "errors"])
for seed in range(5):
    if str(seed) in done: continue
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    L_, d = [], 16
    for _ in range(3): L_ += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L_, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(0, STEPS + 1):
        if step > 0:
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): err = ((net(Xte) > 0).float() != Yte).numpy()
            for b in range(1, 8):
                Lte = LS[b][te]
                for l in range(0, b + 1):
                    m = Lte == l
                    if m.sum(): w.writerow([seed, step, b, l, int(m.sum()), int(err[m, b].sum())])
            fh.flush()
    print("seed", seed, "done", flush=True)
