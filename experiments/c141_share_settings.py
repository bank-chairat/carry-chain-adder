import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8; BS = 256; K = 500; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CFG = [("W64", 64, 0, 1.0, 1e-3), ("W128", 128, 0, 1.0, 1e-3), ("W512", 512, 0, 1.0, 1e-3), ("frozen1024", 128, 1024, 1.0, 1e-3), ("CS2", 128, 0, 2.0, 1e-3), ("CS4", 128, 0, 4.0, 1e-3), ("lr3e-4", 128, 0, 1.0, 3e-4), ("lr1e-2", 128, 0, 1.0, 1e-2)]
SEL = os.environ.get("CFGS", "").split()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
fp = f"{OUT}/share.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["cfg", "seed", "share_25_500", "share_25_100", "share_250_500"])
for name, W, FW, CS, LR in CFG:
    if SEL and name not in SEL: continue
    for seed in range(5):
        perm = np.random.default_rng(500 + seed).permutation(len(A)); tr = perm[:len(A) // 2]; Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr])
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        if FW:
            frozen = nn.Sequential(nn.Linear(16, FW), nn.ReLU())
            for p in frozen.parameters(): p.requires_grad_(False)
            d = FW
        else: frozen = nn.Identity(); d = 16
        L1 = nn.Sequential(nn.Linear(d, W), nn.ReLU()); L2 = nn.Sequential(nn.Linear(W, W), nn.ReLU()); L3 = nn.Sequential(nn.Linear(W, W), nn.ReLU()); head = nn.Linear(W, n + 1)
        if CS != 1.0:
            with torch.no_grad():
                for p in [p for m in (L1, L2, L3, head) for p in m.parameters()]: p.mul_(CS)
        shared = [p for m in (L1, L2, L3) for p in m.parameters()]; opt = torch.optim.Adam(shared + list(head.parameters()), lr=LR); sh = []
        for step in range(1, K + 1):
            i = rng.integers(0, len(tr), BS); per = nn.functional.binary_cross_entropy_with_logits(head(L3(L2(L1(frozen(Xtr[i]))))), Ytr[i], reduction="none").mean(0)
            if step % 25 == 0:
                g4 = torch.autograd.grad(per[4], shared, retain_graph=True); ga = torch.autograd.grad(per.sum(), shared, retain_graph=True)
                sh.append(float(torch.sqrt(sum((g ** 2).sum() for g in g4)) / torch.sqrt(sum((g ** 2).sum() for g in ga))))
            loss = per.sum(); opt.zero_grad(); loss.backward(); opt.step()
        sh = np.array(sh); w.writerow([name, seed, round(sh.mean(), 4), round(sh[:4].mean(), 4), round(sh[9:].mean(), 4)]); fh.flush(); print(name, seed, round(sh.mean(), 3), flush=True)
