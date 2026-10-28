# C-130 (R2/R1): how sensitive is the joint bit-4 time to small changes of the initial first layer? Gaussian noise of relative size sigma (times the parameter's initial standard deviation) is added to layer-1 weights and biases of a C-83 initialisation; training as C-83.
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 9000
INITS = [int(x) for x in os.environ["INITS"].split()]; SIGMAS = [float(x) for x in os.environ.get("SIGMAS", "0 0.001 0.01 0.03 0.1 0.3").split()]; DRAWS = int(os.environ.get("DRAWS", "3"))
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te]); t4 = 2 * Y[te][:, 4] - 1
SD = 0.25 / np.sqrt(3)            # std of PyTorch's default U(-1/sqrt(16), 1/sqrt(16)) for layer 1 (fan-in 16)
fp = f"{OUT}/perturb.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["s", "sigma", "draw", "t4", "corr125"])
for s in INITS:
    for sigma in SIGMAS:
        for draw in (range(DRAWS) if sigma > 0 else [0]):
            torch.manual_seed(10000 + s); L, d = [], 16
            for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
            net = nn.Sequential(*L, nn.Linear(d, n + 1))
            if sigma > 0:
                g = np.random.default_rng(100000 * s + 1000 * draw + int(round(sigma * 1e4)))
                with torch.no_grad():
                    net[0].weight.add_(torch.tensor(g.normal(0, sigma * SD, net[0].weight.shape), dtype=torch.float32))
                    net[0].bias.add_(torch.tensor(g.normal(0, sigma * SD, net[0].bias.shape), dtype=torch.float32))
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
            w.writerow([s, sigma, draw, first if first is not None else -1, round(c125, 4)]); fh.flush(); print(s, sigma, draw, first, round(c125, 3), flush=True)
