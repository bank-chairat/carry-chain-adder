# C-116 (W4/W5): does confining the other outputs to the last hidden layer remove the width and frozen-layer dependence of the joint delay? Plain Adam (shared moments).
# arms: joint (all outputs train everything), conf3 (other outputs' gradient reaches only layer 3 and the heads), alone (other outputs' gradient reaches only the heads).
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = int(os.environ.get("CAP", "10000"))
W = int(os.environ.get("W", "128")); CS = float(os.environ.get("CS", "1")); LR = float(os.environ.get("LR", "1e-3")); OPT = os.environ.get("OPT", "adam"); FW = int(os.environ.get("FW", "0")); SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; ARMS = os.environ.get("ARMS", "alone conf3 joint").split()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); SUMV = (A + B).astype(np.float64)
def r2_on(h, s):
    hc = h - h.mean(0); sc = (s - s.mean()) / s.std(); beta = (hc * sc[:, None]).mean(0); return float((beta ** 2).sum() / hc.var(0).sum())
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False
fp = f"{OUT}/confine.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["W", "FW", "CS", "seed", "arm", "t4", "L2_R2S250", "L3_R2S250"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    for arm in ARMS:
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
        params = [p for m in (L1, L2, L3, head) for p in m.parameters()]; opt = torch.optim.SGD(params, lr=LR, momentum=0.9) if OPT == "sgd" else torch.optim.Adam(params, lr=LR)
        def fwd(x):
            hl = L2(L1(frozen(x))); h3 = L3(hl); z = head(h3)
            if arm == "joint": return z
            zo = head(L3(hl.detach())) if arm == "conf3" else head(h3.detach())
            return torch.where(m_oth, zo, z)
        first = None; prev_ok = False; prev_t = 0; r2 = [float("nan")] * 2
        for step in range(1, CAP + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step == 250:
                with torch.no_grad(): h2 = L2(L1(frozen(Xtr[:8192]))); h3 = L3(h2)
                r2 = [r2_on(h2.numpy().astype(np.float64), SUMV[tr][:8192]), r2_on(h3.numpy().astype(np.float64), SUMV[tr][:8192])]
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good and first is None: first = prev_t
                prev_ok = good; prev_t = step
                if first is not None and step >= 250: break
        w.writerow([W, FW, CS, seed, arm, first if first is not None else -1, round(r2[0], 4), round(r2[1], 4)]); fh.flush(); print(W, FW, seed, arm, first, [round(x, 3) for x in r2], flush=True)
