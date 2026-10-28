import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 100; BS = 256; CAP = 12000
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
Xall = torch.tensor(X)
def select(net):
    W1 = net[0].weight.detach().numpy(); W2 = net[2].weight.detach().numpy(); out = np.linalg.norm(W2, axis=0); keep = np.where(out >= np.median(out))[0]; G = []
    for u in keep:
        w = (np.abs(W1[u, :8]) + np.abs(W1[u, 8:])) / 2; m = w.max()
        act = np.where(w >= 0.1 * m)[0]
        if m > 0 and len(act) >= 3 and np.polyfit(act, np.log2(w[act]), 1)[0] >= 0.5: G.append(int(u))
    return keep, G
def accs(net, te, ablate, mean1):
    with torch.no_grad():
        h = net[1](net[0](Xall[te]))
        if len(ablate): h[:, ablate] = torch.tensor(mean1[ablate])
        o = net[6](net[5](net[4](net[3](net[2](h)))))
        return ((o > 0).float().numpy() == S[te]).mean(0)
w = csv.writer(open(f"{OUT}/ablate_n8.csv", "w", newline="")); w.writerow(["seed", "group", "nG"] + [f"drop{b}" for b in range(n + 1)])
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(S[tr]), torch.tensor(S[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(1, CAP + 1):
        i = rng.integers(0, len(tr), BS)
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xall[te]) > 0).float() == Yte).float().mean(0).numpy()
            if (acc >= 0.99).all(): break
    with torch.no_grad(): mean1 = net[1](net[0](Xall)).mean(0).numpy()
    base = accs(net, te, [], mean1); keep, G = select(net); nG = len(G); rest = [int(u) for u in keep if int(u) not in G]
    w.writerow([seed, "G", nG] + [round(float(base[b] - a), 4) for b, a in enumerate(accs(net, te, G, mean1))])
    W2 = net[2].weight.detach().numpy(); onorm = np.linalg.norm(W2, axis=0); T = sorted(rest, key=lambda u: -onorm[u])[:nG]
    w.writerow([seed, "T", nG] + [round(float(base[b] - a), 4) for b, a in enumerate(accs(net, te, T, mean1))])
    rr = np.random.default_rng(seed); drops = np.zeros(n + 1)
    for _ in range(10): drops += base - accs(net, te, list(rr.choice(rest, nG, replace=False)), mean1)
    w.writerow([seed, "R", nG] + [round(float(x), 4) for x in drops / 10]); print(seed, "nG", nG, flush=True)
