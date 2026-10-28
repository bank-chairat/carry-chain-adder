# C-115 (W1, prospective): does the extra recovery time of the 500-step joint body scale as 1/(layer-2 learning-rate multiplier)?
# C-111 continuation (head reset, fresh Adam, bit-4-only body gradient, fixed continuation stream) from the all9_K500 body ('base') or from the same body with layer-2 directions restored ('D_L2', the floor), with the Adam learning rate of layer 2 multiplied by m.
import sys, os, copy, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 3500; K = 500; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False
def make():
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L), nn.Linear(d, n + 1)
def fwd(body, head, x, block):
    h = body(x); o_full = head(h); o_oth = head(h.detach()) if block else o_full
    return torch.where(m_oth, o_oth, o_full)
fp = f"{OUT}/rotation.csv"; fh = open(fp, "a", newline=""); w = csv.writer(fh)
if os.path.getsize(fp) == 0: w.writerow(["seed", "start", "m", "cost"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    body, head = make(); body0 = copy.deepcopy(body); init_head = copy.deepcopy(head.state_dict())
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
    for step in range(1, K + 1):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(head(body(Xtr[i])), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    bsd = copy.deepcopy(body.state_dict())
    for start in ("base", "D_L2"):
        for mm in (0.5, 1.0, 2.0, 4.0):
            b2, _ = make(); b2.load_state_dict(copy.deepcopy(bsd))
            if start == "D_L2":
                with torch.no_grad():
                    W, W0, b, b0 = b2[2].weight, body0[2].weight, b2[2].bias, body0[2].bias
                    r = W.norm(dim=1) / W0.norm(dim=1); W.copy_(W0 * r[:, None]); b.copy_(b0 * r)
            h2 = nn.Linear(128, n + 1); h2.load_state_dict(init_head)
            groups = [dict(params=[b2[0].weight, b2[0].bias, b2[4].weight, b2[4].bias] + list(h2.parameters()), lr=1e-3), dict(params=[b2[2].weight, b2[2].bias], lr=1e-3 * mm)]
            o2 = torch.optim.Adam(groups); r2 = np.random.default_rng(seed * 100 + 77); first = None; prev_ok = False; prev_t = 0
            for step in range(1, CAP + 1):
                i = r2.integers(0, len(tr), BS)
                loss = nn.functional.binary_cross_entropy_with_logits(fwd(b2, h2, Xtr[i], True), Ytr[i], reduction="none").sum(1).mean(); o2.zero_grad(); loss.backward(); o2.step()
                if step % EV == 0:
                    with torch.no_grad(): acc = ((fwd(b2, h2, Xte, True)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                    good = acc >= 0.99
                    if prev_ok and good: first = prev_t; break
                    prev_ok = good; prev_t = step
            w.writerow([seed, start, mm, first if first is not None else -1]); fh.flush(); print(seed, start, mm, first, flush=True)
