"""C-191: root mean square of the last hidden layer during training (steps 0, 50, 100, 250, 500, 1,000), arms alone / joint / joint with fixed-norm rows (C-116 recipe)."""
import os, sys, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel(); X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
class FixedNormLinear(nn.Module):
    def __init__(self, lin):
        super().__init__(); self.v = nn.Parameter(lin.weight.detach().clone()); self.bias = nn.Parameter(lin.bias.detach().clone()); self.register_buffer("c", lin.weight.detach().norm(dim=1, keepdim=True).clone())
    def forward(self, x): return nn.functional.linear(x, self.c * self.v / self.v.norm(dim=1, keepdim=True), self.bias)
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False; REC = (0, 50, 100, 250, 500, 1000)
for arm in ("alone", "joint", "joint_fixnorm"):
    for seed in range(5):
        perm = np.random.default_rng(500 + seed).permutation(len(A)); tr = perm[:len(A) // 2]; Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr])
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        lins = [nn.Linear(16, 128), nn.Linear(128, 128), nn.Linear(128, 128)]
        if arm == "joint_fixnorm": lins = [FixedNormLinear(l) for l in lins]
        L1, L2, L3 = [nn.Sequential(l, nn.ReLU()) for l in lins]; head = nn.Linear(128, n + 1)
        opt = torch.optim.Adam([p for m in (L1, L2, L3, head) for p in m.parameters()], lr=1e-3); out = []
        def fwd(x):
            h3 = L3(L2(L1(x))); z = head(h3)
            return z if arm != "alone" else torch.where(m_oth, head(h3.detach()), z)
        for step in range(0, 1001):
            if step > 0:
                i = rng.integers(0, len(tr), 256); loss = nn.functional.binary_cross_entropy_with_logits(fwd(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step in REC:
                with torch.no_grad(): out.append(float(torch.sqrt((L3(L2(L1(Xtr[:8192]))) ** 2).mean())))
        print(arm, seed, " ".join(f"{v:.3f}" for v in out), flush=True)
