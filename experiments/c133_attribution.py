# C-133 (R2): which direction of the initial first layer creates bit 4's head start? Differentiate J = correlation(bit-4 logit, target) on the held-out set after K = 125 joint Adam steps
# with respect to the initial layer-1 weights and biases, through the unrolled training (float64, the C-83 recipe: split 500, batch stream 777).
import sys, os, json, numpy as np, torch, torch.nn as nn
torch.set_num_threads(2); torch.set_default_dtype(torch.float64); n = 8; BS = 256; K = int(os.environ.get("K", "125")); LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
INITS = [int(x) for x in os.environ["INITS"].split()]; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float64)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]); t4 = torch.tensor(2 * Y[te][:, 4] - 1)
def init_params(s):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128, dtype=torch.float32), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1, dtype=torch.float32))
    return [p.detach().double().clone() for p in net.parameters()]
def f(P, x):
    h = torch.relu(x @ P[0].T + P[1]); h = torch.relu(h @ P[2].T + P[3]); h = torch.relu(h @ P[4].T + P[5]); return h @ P[6].T + P[7]
def J_of(P0):
    P = list(P0); m = [torch.zeros_like(p) for p in P]; v = [torch.zeros_like(p) for p in P]; rng = np.random.default_rng(777)
    for t in range(1, K + 1):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(f(P, Xtr[i]), ytr[i], reduction="none").sum(1).mean()
        g = torch.autograd.grad(loss, P, create_graph=True); c1 = 1 - B1 ** t; c2 = 1 - B2 ** t
        m = [B1 * mm + (1 - B1) * gg for mm, gg in zip(m, g)]; v = [B2 * vv + (1 - B2) * gg * gg for vv, gg in zip(v, g)]
        P = [p - LR * (mm / c1) / (torch.sqrt(vv / c2 + 1e-30) + EPS) for p, mm, vv in zip(P, m, v)]
    z = f(P, Xte)[:, 4]; zc = z - z.mean(); tc = t4 - t4.mean()
    return (zc * tc).sum() / torch.sqrt((zc * zc).sum() * (tc * tc).sum())
for s in INITS:
    P0 = init_params(s); W1 = P0[0].clone().requires_grad_(True); b1 = P0[1].clone().requires_grad_(True)
    rest = [q.clone().requires_grad_(True) for q in P0[2:]]; J = J_of([W1, b1] + rest); gW, gb = torch.autograd.grad(J, [W1, b1])
    np.savez(f"{OUT}/grad_s{s}.npz", gW=gW.numpy(), gb=gb.numpy(), J=float(J))
    print(s, "J(step %d) = %.4f" % (K, float(J)), "|grad| W1 %.3e b1 %.3e" % (float(gW.norm()), float(gb.norm())), flush=True)
