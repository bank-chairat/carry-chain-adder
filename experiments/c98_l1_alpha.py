# C-98 step 1: gradient L1 of f_4 at init for the rescaled-but-functionally-identical nets (W=128), no training.
import numpy as np, torch, torch.nn as nn, json
torch.set_num_threads(1); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = torch.tensor(np.concatenate([bits(A, n), bits(B, n)], 1)); idx = np.random.default_rng(5).choice(len(X), 256, replace=False)
def make(seed, alpha):
    torch.manual_seed(seed * 100 + 4)
    net = nn.Sequential(nn.Linear(16,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,9))
    with torch.no_grad():
        net[0].weight.mul_(alpha); net[0].bias.mul_(alpha); net[2].weight.div_(alpha)
    return net
if __name__ == "__main__":
    res = {}
    for a in (0.25, 0.5, 1, 2, 4):
        v = []
        for s in range(5):
            net = make(s, a); ps = list(net.parameters()); tot = 0.0
            for i in idx: tot += sum(float(g.abs().sum()) for g in torch.autograd.grad(net(X[i:i+1])[0, 4], ps))
            v.append(tot / len(idx))
        base = [make(s, 1) for s in range(5)]
        with torch.no_grad(): dev = max(float((make(s, a)(X[idx]) - base[s](X[idx])).abs().max()) for s in range(5))
        res[a] = v; print("alpha", a, "L1 median %.1f" % np.median(v), "per-seed", [round(x, 1) for x in v], "max |output diff vs alpha=1| %.2e" % dev, flush=True)
    json.dump(res, open("c98_l1_alpha.json", "w"))
