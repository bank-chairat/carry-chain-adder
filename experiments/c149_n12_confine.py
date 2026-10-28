# C-149 / C-170: C-116 arms (env NORM=ln adds a LayerNorm before each ReLU, C-170) (alone, conf3, joint) at n = 12 with the C-48 data recipe (300,000 sampled training pairs, 20,000 held-out pairs, 13 outputs).
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 12; EV = 100; BS = 256; STEPS = 20000; ARMS = os.environ.get("ARMS", "alone conf3 joint").split(); SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
_r = np.random.default_rng(12345); _ids = np.unique(_r.integers(0, 1 << (2 * n), 330000))[:320000]; _r.shuffle(_ids)
A = _ids >> n; B = _ids & ((1 << n) - 1)
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
tr, te = np.arange(300000), np.arange(300000, 320000)
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False
fp = f"{OUT}/n12confine.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["seed", "arm", "time_bit4", "acc_bit4"])
Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(S[tr]), torch.tensor(S[te])
for seed in SEEDS:
    for arm in ARMS:
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        NL = (lambda: [nn.LayerNorm(128)]) if os.environ.get('NORM', 'none') == 'ln' else (lambda: [])
        L1 = nn.Sequential(nn.Linear(2 * n, 128), *NL(), nn.ReLU()); L2 = nn.Sequential(nn.Linear(128, 128), *NL(), nn.ReLU()); L3 = nn.Sequential(nn.Linear(128, 128), *NL(), nn.ReLU()); head = nn.Linear(128, n + 1)
        opt = torch.optim.Adam([p for mm in (L1, L2, L3, head) for p in mm.parameters()], lr=1e-3)
        def fwd(x):
            hl = L2(L1(x)); h3 = L3(hl); z = head(h3)
            if arm == "joint": return z
            zo = head(L3(hl.detach())) if arm == "conf3" else head(h3.detach())
            return torch.where(m_oth, zo, z)
        first = -1; prev_ok = False; prev_t = 0; acc = 0.0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if first < 0 and prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([seed, arm, first, round(acc, 4)]); fh.flush(); print(seed, arm, first, round(acc, 4), flush=True)
