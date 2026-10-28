import json, numpy as np, torch, torch.nn as nn
torch.set_num_threads(2); n = 8; BS = 256
on = {r['s']: r for r in json.load(open('c125_onsets.json'))}
pick = sorted(on, key=lambda s: on[s]['t_on'])[:6]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]); t4 = 2 * Y[te][:, 4] - 1; Xall = torch.tensor(X)
def corr(z): return float(np.corrcoef(z.numpy().astype(np.float64), t4)[0, 1])
def pct(v, idx): return [float((v < v[i]).mean() * 100) for i in idx]
for s in pick:
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); W0 = net[0].weight.detach().clone().numpy(); b0 = net[0].bias.detach().clone().numpy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777); K = on[s]['t_on'] + 100
    for step in range(1, K + 1):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Jb = corr(net(Xte)[:, 4]); out = {}
        for li, idx in enumerate((1, 3, 5)):
            mean = net[:idx + 1](Xtr).mean(0); drops = np.zeros(128)
            for u in range(128):
                h = net[:idx + 1](Xte).clone(); h[:, u] = mean[u]; drops[u] = Jb - corr(net[idx + 1:](h)[:, 4])
            pos = np.clip(drops, 0, None); top = np.argsort(-drops)[:5]; out[li] = (top, pos[top].sum() / max(pos.sum(), 1e-12))
        top1 = out[0][0]; W1 = W0.astype(np.float64)
        p4share = (W1[:, 4] ** 2 + W1[:, 12] ** 2) / (W1 ** 2).sum(1); anti = np.abs(W1[:, 4] - W1[:, 12]); act = ((X @ W1.T + b0) > 0).mean(0); half = -np.abs(act - 0.5)
        moved = np.linalg.norm(net[0].weight.numpy() - W0, axis=1) / np.linalg.norm(W0, axis=1)
    print(f"s {s} t_on {on[s]['t_on']} step {K} J {Jb:.3f} | top-5 share of positive drops: L1 {out[0][1]:.2f} L2 {out[1][1]:.2f} L3 {out[2][1]:.2f} | top L1 units' init percentiles: pos4-share {np.median(pct(p4share, top1)):.0f}, |a4-b4| {np.median(pct(anti, top1)):.0f}, activity~1/2 {np.median(pct(half, top1)):.0f}; moved {np.median(pct(moved, top1)):.0f}", flush=True)
