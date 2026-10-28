import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1)
T = {"p4": ((A ^ B) >> 4) & 1, "c4": (((A & 15) + (B & 15)) >> 4) & 1, "p3": ((A ^ B) >> 3) & 1}
w = csv.writer(open(f"{OUT}/init_probe.csv", "w", newline="")); w.writerow(["W", "seed", "target", "layer", "probe_acc"])
for WS in [(64,512,512),(512,64,64),(64,64,512),(512,512,64),(128,128,128)]:
    for seed in range(5):
        perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
        pr = np.random.default_rng(700 + seed); ptr = pr.choice(tr, 20000, replace=False); pte = pr.choice(te, 20000, replace=False)
        torch.manual_seed(seed * 100 + 4); layers = []; d = 2 * n
        for k in range(3): layers.append(nn.Sequential(nn.Linear(d, WS[k]), nn.ReLU())); d = WS[k]
        head = nn.Linear(d, n + 1)     # created so that the random stream matches the training runs
        with torch.no_grad():
            for l in layers:
                for p in l.parameters(): p.mul_(1.0)
            def acts(idx):
                x = torch.tensor(X[idx]); hs = []
                for l in layers: x = l(x); hs.append(x.numpy())
                return hs
            htr, hte = acts(ptr), acts(pte)
        for L in range(3):
            m_, s_ = htr[L].mean(0), htr[L].std(0) + 1e-6
            a_ = np.c_[(htr[L] - m_) / s_, np.ones(len(ptr))]; b_ = np.c_[(hte[L] - m_) / s_, np.ones(len(pte))]
            G = np.linalg.inv(a_.T @ a_ + 1.0 * np.eye(a_.shape[1]))
            for name, y in T.items():
                wv = G @ (a_.T @ (2.0 * y[ptr] - 1.0)); pred = (b_ @ wv > 0).astype(int)
                w.writerow(["-".join(map(str,WS)), seed, name, L + 1, round(float((pred == y[pte]).mean()), 4)])
