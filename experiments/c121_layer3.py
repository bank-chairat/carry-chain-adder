# C-121: why does keeping the other outputs out of layer 3 (but not out of layers 1-2) slow bit 4 so much (C-114)?
# MODE=sep: C-73/C-114 optimiser (each output its own Adam moments and full step in the shared layers); arms none, L3.
# MODE=plain: ordinary Adam (shared moments); arms joint, blockL3 (other outputs' gradient is not applied to layer-3 parameters but still flows through layer 3 into layers 1-2).
# At step 250: gradient coherence of the other outputs' summed gradient in layers 1 and 2 over 8 extra minibatches (a separate random stream, training unaffected).
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; STEPS = 6000; LR = 1e-3; B1, B2, EPS = 0.9, 0.999, 1e-8
MODE = os.environ["MODE"]; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; ARMS = os.environ["ARMS"].split()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
oth = [k for k in range(n + 1) if k != 4]
fp = f"{OUT}/layer3.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["mode", "seed", "arm", "t4", "coh_oth_L1", "coh_oth_L2", "coh_b4_L2", "norm_oth_over_b4_L2", "relchange_L1_500", "relchange_L2_500"])
def coherence(gs):
    G = torch.stack([g.flatten() for g in gs]); m = G.mean(0); return float((m @ m) / (G * G).sum(1).mean())
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
    for arm in ARMS:
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        layers = []; d = 2 * n
        for _ in range(3): layers.append(nn.Sequential(nn.Linear(d, 128), nn.ReLU())); d = 128
        head = nn.Linear(d, n + 1); trunk = [p for l in layers for p in l.parameters()]; hp = list(head.parameters())
        W10, W20 = layers[0][0].weight.detach().clone(), layers[1][0].weight.detach().clone()
        def fwd(x):
            for l in layers: x = l(x)
            return head(x)
        if MODE == "sep":
            opt_h = torch.optim.Adam(hp, lr=LR); K = n + 1
            m = [[torch.zeros_like(p) for p in trunk] for _ in range(K)]; v = [[torch.zeros_like(p) for p in trunk] for _ in range(K)]
        else:
            opt = torch.optim.Adam(trunk + hp, lr=LR)
        L3p = set(id(p) for p in layers[2].parameters())
        first = None; prev_ok = False; prev_t = 0; diag = [float("nan")] * 4; rel = [float("nan")] * 2
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS); o = fwd(Xtr[i]); bce = nn.functional.binary_cross_entropy_with_logits(o, Ytr[i], reduction="none").mean(0)
            if MODE == "sep":
                upd = [torch.zeros_like(p) for p in trunk]; hg = [torch.zeros_like(p) for p in hp]; c1 = 1 - B1 ** step; c2 = 1 - B2 ** step
                for k in range(K):
                    g = torch.autograd.grad(bce[k], trunk + hp, retain_graph=(k < K - 1))
                    for j, p in enumerate(trunk):
                        gj = g[j]; m[k][j].mul_(B1).add_((1 - B1) * gj); v[k][j].mul_(B2).add_((1 - B2) * gj * gj)
                        if arm == "L3" and k != 4 and id(p) in L3p: continue
                        upd[j] += (m[k][j] / c1) / (torch.sqrt(v[k][j] / c2) + EPS)
                    for j in range(len(hp)): hg[j] += g[len(trunk) + j]
                with torch.no_grad():
                    for j, p in enumerate(trunk): p.sub_(LR * upd[j])
                for p, gg in zip(hp, hg): p.grad = gg
                opt_h.step()
            else:
                g4 = torch.autograd.grad(bce[4], trunk + hp, retain_graph=True); go = torch.autograd.grad(bce[oth].sum(), trunk + hp)
                for p, a, b in zip(trunk + hp, g4, go): p.grad = a + (torch.zeros_like(b) if (arm == "blockL3" and id(p) in L3p) else b)
                opt.step()
            if step == 250:
                r3 = np.random.default_rng(999); go1, go2, g42 = [], [], []
                for _ in range(8):
                    j = r3.integers(0, len(tr), BS); oo = fwd(Xtr[j]); bb = nn.functional.binary_cross_entropy_with_logits(oo, Ytr[j], reduction="none").mean(0)
                    a1, a2 = torch.autograd.grad(bb[oth].sum(), [layers[0][0].weight, layers[1][0].weight], retain_graph=True); (b2,) = torch.autograd.grad(bb[4], [layers[1][0].weight])
                    go1.append(a1); go2.append(a2); g42.append(b2)
                diag = [coherence(go1), coherence(go2), coherence(g42), float(torch.stack(go2).norm() / torch.stack(g42).norm())]
            if step == 500:
                rel = [float((layers[0][0].weight.detach() - W10).norm() / W10.norm()), float((layers[1][0].weight.detach() - W20).norm() / W20.norm())]
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good and first is None: first = prev_t
                prev_ok = good; prev_t = step
                if first is not None and step >= 500: break
        w.writerow([MODE, seed, arm, first if first is not None else -1] + [round(x, 4) for x in diag + rel]); fh.flush(); print(MODE, seed, arm, first, [round(x, 3) for x in diag + rel], flush=True)
