import numpy as np, torch, torch.nn as nn, sys
torch.set_default_dtype(torch.float64); torch.set_num_threads(2); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float64)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
def build(seed, alpha):
    torch.manual_seed(seed * 100 + 4)
    net = nn.Sequential(nn.Linear(16,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,9))
    with torch.no_grad(): net[0].weight.mul_(alpha); net[0].bias.mul_(alpha); net[2].weight.div_(alpha)
    return net
def run(seed, alpha, comp, steps=300, eta=1e-3, eps=1e-8):
    net = build(seed, alpha); L = [net[0], net[2], net[4], net[6]]
    if comp:
        groups = [dict(params=[net[0].weight, net[0].bias], lr=eta * alpha, eps=eps / alpha), dict(params=[net[2].weight], lr=eta / alpha, eps=eps * alpha),
                  dict(params=[net[2].bias, net[4].weight, net[4].bias, net[6].weight, net[6].bias], lr=eta, eps=eps)]
    else: groups = [dict(params=list(net.parameters()), lr=eta, eps=eps)]
    opt = torch.optim.Adam(groups)
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr]); Xte = torch.tensor(X[te][:4096]); rng = np.random.default_rng(seed * 100 + 4); out = {}
    for s in range(1, steps + 1):
        i = rng.integers(0, len(tr), 256)
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if s in (100, 300):
            with torch.no_grad(): out[s] = net(Xte).clone()
    return out
ref = run(0, 1, False); comp = run(0, 4, True); unc = run(0, 4, False)
for s in (100, 300): print("step", s, "max |logit diff| compensated vs ref %.3e   uncompensated vs ref %.3e" % (float((comp[s] - ref[s]).abs().max()), float((unc[s] - ref[s]).abs().max())))
