# C-113b: bit 4 trained together with four companion targets of different kinds; does a companion that builds a code of the numeric sum S slow bit 4 more than one that builds a code of an unrelated direction?
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 6000; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); S = (A + B).astype(np.float64)
def r2_on(h, s):
    hc = h - h.mean(0); sc = (s - s.mean()) / s.std(); beta = (hc * sc[:, None]).mean(0); return float((beta ** 2).sum() / hc.var(0).sum())
fp = f"{OUT}/companions.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["seed", "cond", "t4", "L2_R2_S_at250", "L2_R2_proj_at250", "companion_info"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    g = np.random.default_rng(3000 + seed)
    qs = g.uniform(0.2, 0.8, 4); tS = np.quantile(S, qs)                                  # thresholds of S at random quantiles
    Wr = g.uniform(0, 1, (4, 16)); P = X @ Wr.T; tR = np.array([np.quantile(P[:, k], qs[k]) for k in range(4)])   # random positive directions, same quantiles
    comp = {"alone": None, "upper4": Y[:, 5:9], "Sthr4": (S[:, None] >= tS[None, :]).astype(np.float32), "Rthr4": (P >= tR[None, :]).astype(np.float32)}
    for cond, C in comp.items():
        Yc = Y[:, [4]] if C is None else np.concatenate([Y[:, [4]], C], 1)
        Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Yc[tr]), torch.tensor(Yc[te])
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        L, d = [], 16
        for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        net = nn.Sequential(*L, nn.Linear(d, Yc.shape[1])); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        first = None; prev_ok = False; prev_t = 0; r2s = r2p = float("nan")
        for step in range(1, CAP + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step == 250:
                with torch.no_grad(): h2 = net[:4](Xtr[:8192]).numpy().astype(np.float64)
                r2s = r2_on(h2, S[tr][:8192]); r2p = r2_on(h2, P[tr][:8192, 0])
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte)[:, 0] > 0).float() == Yte[:, 0]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good and first is None: first = prev_t
                prev_ok = good; prev_t = step
                if first is not None and step >= 250: break
        info = "" if C is None else f"balance {np.round(C.mean(0), 2).tolist()}"
        w.writerow([seed, cond, first if first is not None else -1, round(r2s, 4), round(r2p, 4), info]); fh.flush(); print(seed, cond, first, round(r2s, 3), round(r2p, 3), info, flush=True)
