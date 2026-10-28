# C-112: is the cost of the 500-step joint body due to its weight-norm growth or to its feature geometry (directions / collapse)?
# Arms on the all9_K500 body (seeds 0-4), then the C-111 continuation (head reset, fresh Adam, bit-4-only body gradient, fixed continuation stream).
import sys, os, copy, csv, json, numpy as np, torch, torch.nn as nn
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
@torch.no_grad()
def arm(tag, body, body0):
    W = [body[i].weight for i in (0, 2, 4)]; b = [body[i].bias for i in (0, 2, 4)]; W0 = [body0[i].weight for i in (0, 2, 4)]; b0 = [body0[i].bias for i in (0, 2, 4)]
    c = [float(W0[l].norm() / W[l].norm()) for l in range(3)]
    if tag == "base": pass
    elif tag == "N_all":            # each layer's weights back to the initial Frobenius norm; biases scaled by the cumulative factor so every ReLU on/off pattern is unchanged
        cum = 1.0
        for l in range(3): cum *= c[l]; W[l].mul_(c[l]); b[l].mul_(cum)
    elif tag == "N_L2":             # only layer 2 back to its initial norm (W2, b2 and b3 scaled, patterns unchanged)
        W[1].mul_(c[1]); b[1].mul_(c[1]); b[2].mul_(c[1])
    elif tag == "D_L2":             # layer-2 directions back to the initial ones, each row keeping its trained norm (bias scaled with the row)
        r = W[1].norm(dim=1) / W0[1].norm(dim=1); W[1].copy_(W0[1] * r[:, None]); b[1].copy_(b0[1] * r)
    elif tag == "R_L2":             # layer 2 reset to its initial weights and bias (anchor, like C-106 cur101)
        W[1].copy_(W0[1]); b[1].copy_(b0[1])
    return c
fp = f"{OUT}/norm_geom.csv"; fh = open(fp, "a", newline=""); w = csv.writer(fh)
if os.path.getsize(fp) == 0: w.writerow(["seed", "arm", "cost", "c1", "c2", "c3"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    body, head = make(); body0 = copy.deepcopy(body); init_head = copy.deepcopy(head.state_dict())
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
    for step in range(1, K + 1):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(head(body(Xtr[i])), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    bsd = copy.deepcopy(body.state_dict())
    for tag in ("base", "N_all", "N_L2", "D_L2", "R_L2"):
        b2, _ = make(); b2.load_state_dict(copy.deepcopy(bsd)); c = arm(tag, b2, body0)
        h2 = nn.Linear(128, n + 1); h2.load_state_dict(init_head); o2 = torch.optim.Adam(list(b2.parameters()) + list(h2.parameters()), lr=1e-3)
        r2 = np.random.default_rng(seed * 100 + 77); first = None; prev_ok = False; prev_t = 0
        for step in range(1, CAP + 1):
            i = r2.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(b2, h2, Xtr[i], True), Ytr[i], reduction="none").sum(1).mean(); o2.zero_grad(); loss.backward(); o2.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(b2, h2, Xte, True)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([seed, tag, first if first is not None else -1] + [round(x, 4) for x in c]); fh.flush(); print(seed, tag, first, [round(x, 3) for x in c], flush=True)
