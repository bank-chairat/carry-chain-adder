# C-122 (O3): is the width effect the optimiser parametrisation (exactly what duplicating units does to Adam) or genuinely new features?
# arms (each 'joint' and 'alone' = other outputs' gradient reaches only the heads):
#   W128      : standard width 128
#   W128dup4  : width 128 with the Adam settings that a 4x DUPLICATED width-512 network has (derived; see PROGRESS C-122): lr x4 on W2, W3 and head weights, eps x4 on W1, W2, W3 and hidden biases
#   DUP512    : the width-128 initial net duplicated 4 times (Net2Wider-style, no noise), ordinary Adam: must equal W128dup4 up to rounding (theorem check)
#   W512      : standard independently initialised width 512
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 10000; K = 4
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; ARMS = os.environ.get("ARMS", "W128dup4 DUP512").split(); MODES = os.environ.get("MODES", "alone joint").split()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); SUMV = (A + B).astype(np.float64)
def r2_on(h, s):
    hc = h - h.mean(0); sc = (s - s.mean()) / s.std(); beta = (hc * sc[:, None]).mean(0); return float((beta ** 2).sum() / (hc.var(0).sum() + 1e-12))
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False
def base(W):
    return nn.Sequential(nn.Linear(16, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, n + 1))
fp = f"{OUT}/widthdup.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["seed", "arm", "mode", "t4", "L2_R2S250"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    for arm in ARMS:
        for mode in MODES:
            torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
            if arm == "W512": net = base(512); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            elif arm == "W128": net = base(128); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            elif arm == "W128dup4":
                net = base(128); lr, eps = 1e-3, 1e-8
                opt = torch.optim.Adam([dict(params=[net[0].weight], lr=lr, eps=K * eps), dict(params=[net[0].bias, net[2].bias, net[4].bias], lr=lr, eps=K * eps),
                                        dict(params=[net[2].weight, net[4].weight], lr=K * lr, eps=K * eps), dict(params=[net[6].weight], lr=K * lr, eps=eps), dict(params=[net[6].bias], lr=lr, eps=eps)])
            elif arm == "DUP512":
                small = base(128); net = base(512)
                with torch.no_grad():
                    net[0].weight.copy_(small[0].weight.repeat(K, 1)); net[0].bias.copy_(small[0].bias.repeat(K))
                    for i in (2, 4): net[i].weight.copy_(small[i].weight.repeat(K, K) / K); net[i].bias.copy_(small[i].bias.repeat(K))
                    net[6].weight.copy_(small[6].weight.repeat(1, K) / K); net[6].bias.copy_(small[6].bias)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            def fwd(x):
                h = net[:6](x); z = net[6](h)
                if mode == "joint": return z
                return torch.where(m_oth, net[6](h.detach()), z)
            first = None; prev_ok = False; prev_t = 0; r2 = float("nan")
            for step in range(1, CAP + 1):
                i = rng.integers(0, len(tr), BS)
                loss = nn.functional.binary_cross_entropy_with_logits(fwd(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
                if step == 250:
                    with torch.no_grad(): r2 = r2_on(net[:4](Xtr[:8192]).numpy().astype(np.float64), SUMV[tr][:8192])
                if step % EV == 0:
                    with torch.no_grad(): acc = ((fwd(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                    good = acc >= 0.99
                    if prev_ok and good and first is None: first = prev_t
                    prev_ok = good; prev_t = step
                    if first is not None and step >= 250: break
            w.writerow([seed, arm, mode, first if first is not None else -1, round(r2, 4)]); fh.flush(); print(seed, arm, mode, first, round(r2, 3), flush=True)
