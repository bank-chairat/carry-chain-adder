# C-119 (W3): which part of the first layer of a fast initialisation matters? Networks are the fast init except for layer-1 parts taken from the slow init:
# columns of input positions 0-4 (a0..a4, b0..b4; the inputs sum bit 4 depends on), columns of positions 5-7, and the layer-1 biases. Training as C-83/C-118.
import sys, os, csv, itertools, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 9000
PAIRS = [tuple(int(y) for y in x.split(",")) for x in os.environ["PAIRS"].split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr, yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
REL = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]; IRR = [5, 6, 7, 13, 14, 15]
def init_net(s):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
fp = f"{OUT}/l1parts.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["fast", "slow", "combo_rel_irr_bias", "t4"])
for fast, slow in PAIRS:
    nf, ns = init_net(fast).state_dict(), init_net(slow).state_dict()
    for combo in itertools.product("FS", repeat=3):
        net = init_net(fast); sd = {k: v.clone() for k, v in nf.items()}
        if combo[0] == "S": sd["0.weight"][:, REL] = ns["0.weight"][:, REL]
        if combo[1] == "S": sd["0.weight"][:, IRR] = ns["0.weight"][:, IRR]
        if combo[2] == "S": sd["0.bias"] = ns["0.bias"].clone()
        net.load_state_dict(sd); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777)
        first = None; prev_ok = False; prev_t = 0
        for step in range(1, CAP + 1):
            i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte)[:, 4] > 0).float() == yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([fast, slow, "".join(combo), first if first is not None else -1]); fh.flush(); print(fast, slow, "".join(combo), first, flush=True)
