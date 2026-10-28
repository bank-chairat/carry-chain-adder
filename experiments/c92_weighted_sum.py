import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel(); X = np.concatenate([bits(A, n), bits(B, n)], 1); Xall = torch.tensor(X)
M = {k: sum((2.0 ** j) * ((A >> j) & 1) + (2.0 ** j) * ((B >> j) & 1) for j in range(k)) for k in range(2, 9)}
def mknet():
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
def select(net):
    W1 = net[0].weight.detach().numpy(); W2 = net[2].weight.detach().numpy(); out = np.linalg.norm(W2, axis=0); keep = np.where(out >= np.median(out))[0]; G = []
    for u in keep:
        w = (np.abs(W1[u, :8]) + np.abs(W1[u, 8:])) / 2; m = w.max(); act = np.where(w >= 0.1 * m)[0]
        if m > 0 and len(act) >= 3 and np.polyfit(act, np.log2(w[act]), 1)[0] >= 0.5: G.append(int(u))
    return keep, G, out
def best(z):
    r2 = {k: float(np.corrcoef(z, M[k])[0, 1] ** 2) if z.std() > 0 else 0.0 for k in M}; k = max(r2, key=r2.get); return k, r2[k]
w = csv.writer(open(f"{OUT}/weighted_sum_n8.csv", "w", newline="")); w.writerow(["seed", "set", "unit", "best_k", "r2"])
for seed in range(5):
    net = mknet(); net.load_state_dict(torch.load(f"c91_nets/net_s{seed}.pt", weights_only=True)); keep, G, outn = select(net); nG = len(G)
    rest = [int(u) for u in keep if int(u) not in G]; T = sorted(rest, key=lambda u: -outn[u])[:nG]; R = list(np.random.default_rng(seed).choice(rest, nG, replace=False))
    with torch.no_grad(): Z = net[0](Xall).numpy()
    for name, units in (("G", G), ("T", T), ("R", R)):
        for u in units:
            k, r2 = best(Z[:, u].astype(np.float64)); w.writerow([seed, name, int(u), k, round(r2, 4)])
