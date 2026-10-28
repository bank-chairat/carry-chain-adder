# C-118 (W3): which part of the initial weights makes a joint run fast? Build initial networks from a fast and a slow C-83 initialisation, each of layer 1, layer 2, layer 3 and the output layer taken from one of the two; train exactly as C-83 (split seed 500, batch stream 777, Adam 1e-3, batch 256) and record the bit-4 time.
import sys, os, csv, itertools, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 9000
PAIRS = [tuple(int(y) for y in x.split(",")) for x in os.environ["PAIRS"].split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
def init_net(s):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
fp = f"{OUT}/swap.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["fast", "slow", "combo", "t4"])
for fast, slow in PAIRS:
    nf, ns = init_net(fast).state_dict(), init_net(slow).state_dict()
    for combo in itertools.product("FS", repeat=4):        # origin of layer 1, layer 2, layer 3, output layer
        sd = {}
        for li, idx in enumerate((0, 2, 4, 6)):
            src = nf if combo[li] == "F" else ns
            for k in (f"{idx}.weight", f"{idx}.bias"): sd[k] = src[k].clone()
        net = init_net(fast); net.load_state_dict(sd); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777)
        first = None; prev_ok = False; prev_t = 0
        for step in range(1, CAP + 1):
            i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte)[:, 4] > 0).float() == yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([fast, slow, "".join(combo), first if first is not None else -1]); fh.flush(); print(fast, slow, "".join(combo), first, flush=True)
