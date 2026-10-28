# C-101 step 1: train the joint net (as C-39 / C-100: n=8, 3x128, Adam 1e-3, batch 256, seeds 0-2) and save parameters + Adam state at steps 500, 1000, 2000.
import sys, os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); OUT = "c102_ckpt"; os.makedirs(OUT, exist_ok=True); n = 8; BS = 256
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
for seed in range(3):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr = perm[:len(A) // 2]
    Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(1, 301):
        i = rng.integers(0, len(tr), BS)
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step in (50, 100, 200, 300):
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step}, f"{OUT}/s{seed}_t{step}.pt")
    print("seed", seed, "saved", flush=True)
