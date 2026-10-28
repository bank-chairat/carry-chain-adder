import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; BS = 256; CK = [0, 100, 300, 1000]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
w = csv.writer(open(f"{OUT}/grad_cos.csv", "w")); w.writerow(["seed", "step", "other_bit", "cos"])
for seed in range(5):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr = perm[:len(A) // 2]
    Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    params = [p for m in net[:-1] if isinstance(m, nn.Linear) for p in m.parameters()]
    fb = torch.tensor(rng.integers(0, len(tr), 4096))
    def grads():
        out = net(Xtr[fb]); G = []
        for b in range(n + 1):
            l = nn.functional.binary_cross_entropy_with_logits(out[:, b], Ytr[fb][:, b])
            g = torch.autograd.grad(l, params, retain_graph=True); G.append(torch.cat([x.flatten() for x in g]))
        return torch.stack(G)
    for step in range(0, max(CK) + 1):
        if step in CK:
            G = grads(); g4 = G[4]; cs = lambda u, v: float(torch.dot(u, v) / (u.norm() * v.norm() + 1e-12))
            for b in range(n + 1):
                if b != 4: w.writerow([seed, step, b, cs(g4, G[b])])
            w.writerow([seed, step, "sum_others", cs(g4, G.sum(0) - g4)])
        i = rng.integers(0, len(tr), BS)
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    print("seed", seed, flush=True)
