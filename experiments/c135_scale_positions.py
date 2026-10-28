# C-135b: scale layer-1 weight columns of chosen input positions (a and b columns) of a C-83 initialisation, then train with the C-83 recipe.
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8; EV = 50; BS = 256; CAP = 9000; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
JOBS = [x.split(":") for x in os.environ["JOBS"].split()]   # s:tag  with tag in up4down7, down4up7, up5down6
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te]); t4 = 2 * Y[te][:, 4] - 1
fp = f"{OUT}/scale.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["s", "tag", "t4", "corr125"])
for s, tag in JOBS:
    s = int(s); up, down = {"up4down7": (4, 7), "down4up7": (7, 4), "up5down6": (5, 6), "up4": (4, None), "down7": (None, 7), "down4": (None, 4), "up7": (7, None)}[tag]
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1))
    with torch.no_grad():
        if up is not None: net[0].weight[:, [up, up + 8]] *= 1.5
        if down is not None: net[0].weight[:, [down, down + 8]] /= 1.5
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777); first = None; prev_ok = False; prev_t = 0; c125 = float("nan")
    for step in range(1, CAP + 1):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step == 125:
            with torch.no_grad(): z = net(Xte)[:, 4].numpy().astype(np.float64)
            c125 = float(np.corrcoef(z, t4)[0, 1])
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xte)[:, 4] > 0).float() == yte[:, 4]).float().mean().item()
            good = acc >= 0.99
            if prev_ok and good and first is None: first = prev_t
            prev_ok = good; prev_t = step
            if first is not None and step >= 125: break
    w.writerow([s, tag, first if first is not None else -1, round(c125, 4)]); fh.flush(); print(s, tag, first, round(c125, 3), flush=True)
