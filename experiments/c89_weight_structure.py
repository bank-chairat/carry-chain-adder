import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 100; BS = 256; CAP = 12000
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
def structure(net):
    W1 = net[0].weight.detach().numpy(); W2 = net[2].weight.detach().numpy()          # (128,16), (128,128)
    out = np.linalg.norm(W2, axis=0); keep = np.where(out >= np.median(out))[0]; rows = []
    for u in keep:
        w = (np.abs(W1[u, :8]) + np.abs(W1[u, 8:])) / 2; m = w.max()
        if m <= 0: continue
        supp = int((w >= 0.25 * m).sum()); act = np.where(w >= 0.1 * m)[0]
        slope = float(np.polyfit(act, np.log2(w[act]), 1)[0]) if len(act) >= 3 else np.nan
        rows.append((supp, slope))
    return rows
w = csv.writer(open(f"{OUT}/structure_n8.csv", "w", newline="")); w.writerow(["seed", "trained", "support", "slope"])
for seed in range(5):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for r in structure(net): w.writerow([seed, 0, r[0], r[1]])
    for step in range(1, CAP + 1):
        i = rng.integers(0, len(tr), BS)
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xte) > 0).float() == Yte).float().mean(0).numpy()
            if (acc >= 0.99).all(): break
    for r in structure(net): w.writerow([seed, 1, r[0], r[1]])
    print(seed, "steps", step, "min acc %.4f" % acc.min(), flush=True)
