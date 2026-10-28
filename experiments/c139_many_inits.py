import sys, os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8; BS = 256; K = 500
INITS = [int(x) for x in os.environ["INITS"].split()]; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]); t4 = 2 * Y[te][:, 4] - 1
for s in INITS:
    if os.path.exists(f"{OUT}/init_s{s}.npz"): continue
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); W1 = net[0].weight.detach().numpy().copy(); b1 = net[0].bias.detach().numpy().copy()
    shared = list(net[0].parameters()) + list(net[2].parameters()) + list(net[4].parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777); J, share = [], []
    for step in range(0, K + 1):
        if step > 0:
            i = rng.integers(0, len(tr), BS); z = net(Xtr[i]); per = nn.functional.binary_cross_entropy_with_logits(z, ytr[i], reduction="none").mean(0)
            if step % 25 == 0:
                g4 = torch.autograd.grad(per[4], shared, retain_graph=True); ga = torch.autograd.grad(per.sum(), shared, retain_graph=True)
                share.append(float(torch.sqrt(sum((g ** 2).sum() for g in g4)) / torch.sqrt(sum((g ** 2).sum() for g in ga))))
            loss = per.sum(); opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            with torch.no_grad(): zz = net(Xte)[:, 4].numpy().astype(np.float64)
            J.append(float(np.corrcoef(zz, t4)[0, 1]) if zz.std() > 0 else 0.0)
    np.savez(f"{OUT}/init_s{s}.npz", W1=W1, b1=b1, J=np.array(J), share=np.array(share))
    print(s, "J500 %.3f share %.3f" % (J[-1], np.mean(share)), flush=True)
