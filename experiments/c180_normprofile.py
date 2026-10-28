"""C-180 (descriptive): how does the size of each example's last-hidden-layer vector depend on S = a + b after 250 joint steps?
C-116 recipe, joint arm, seeds 0-4; reports the linear correlation, the R^2 of ||h3|| on a 16-bin step function of S (any shape), and the binned profile."""
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel(); X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); SV = (A + B).astype(float)
for seed in range(5):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr = perm[:len(A) // 2]; Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    L1 = nn.Sequential(nn.Linear(16, 128), nn.ReLU()); L2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU()); L3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU()); head = nn.Linear(128, n + 1)
    net = nn.Sequential(L1, L2, L3, head); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(1, 251):
        i = rng.integers(0, len(tr), 256); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): h3 = L3(L2(L1(Xtr[:8192]))).numpy().astype(float)
    s = SV[tr][:8192]; nrm = np.linalg.norm(h3, axis=1); b = np.minimum((s / 32).astype(int), 15)
    means = np.array([nrm[b == k].mean() for k in range(16)]); fit = means[b]; r2 = 1 - ((nrm - fit) ** 2).mean() / nrm.var()
    print(f"seed {seed}: corr(||h3||, S) {np.corrcoef(nrm, s)[0, 1]:+.2f}; R^2 of ||h3|| on 16 S-bins {r2:.2f}; coefficient of variation of ||h3|| {nrm.std() / nrm.mean():.2f}; binned mean (S/32 = 0..15): " + " ".join(f"{m:.1f}" for m in means))
