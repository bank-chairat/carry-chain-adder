# C-111: what does pretraining change in the body, and does it track the later bit-4 recovery cost?
# Pretraining (Adam 1e-3, batch 256) on: all nine sum bits (snapshots K = 100, 250, 500, 1000), the eight other sum bits (K = 250), nine random 4-subset parities (K = 250), or nothing.
# At each snapshot: body diagnostics; then head reset to its initial values, fresh Adam, body trained on bit 4 only (heads train on their own losses), fixed continuation minibatch stream (same for every condition of a seed).
import sys, os, copy, csv, json, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 3500; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False
REL = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]; IRR = [5, 6, 7, 13, 14, 15]          # input columns relevant / irrelevant to sum bit 4
def make():
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L), nn.Linear(d, n + 1)
def fwd(body, head, x, block):
    h = body(x); o_full = head(h); o_oth = head(h.detach()) if block else o_full
    return torch.where(m_oth, o_oth, o_full)
@torch.no_grad()
def diag(body, body0, Xd):
    out = {}; h = Xd
    for li, idx in enumerate((0, 2, 4)):
        h = body[idx + 1](body[idx](h)); W, W0 = body[idx].weight, body0[idx].weight
        act = (h > 0).float(); um = h.mean(0); lm = um.mean()
        hc = h - h.mean(0); sv = torch.linalg.svdvals(hc); p = sv / sv.sum(); lam = sv ** 2
        out[f"L{li+1}_dead"] = float((act.sum(0) == 0).float().mean())
        out[f"L{li+1}_dormant"] = float((um < 0.1 * lm).float().mean())
        out[f"L{li+1}_erank"] = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
        out[f"L{li+1}_pr"] = float(lam.sum() ** 2 / (lam ** 2).sum())
        out[f"L{li+1}_normratio"] = float(W.norm() / W0.norm())
        out[f"L{li+1}_reldelta"] = float((W - W0).norm() / W0.norm())
        out[f"L{li+1}_actfrac"] = float(act.mean())
    W1, W10 = body[0].weight.abs(), body0[0].weight.abs()
    out["L1_irr_over_rel"] = float((W1[:, IRR].sum() / W1[:, REL].sum()) / (W10[:, IRR].sum() / W10[:, REL].sum()))
    return out
fp = f"{OUT}/diag.csv"; fh = open(fp, "a", newline=""); w = None
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    Xd = Xtr[np.random.default_rng(7).choice(len(tr), 8192, replace=False)]
    sub = np.random.default_rng(900 + seed); masks = np.zeros((9, 2 * n), np.float32)
    for j in range(9): masks[j, sub.choice(2 * n, 4, replace=False)] = 1
    Yrand = torch.tensor((X @ masks.T) % 2)[tr]
    snaps = []
    for kind, Ks in (("none", [0]), ("all9", [100, 250, 500, 1000]), ("others8", [250]), ("random9", [250])):
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        body, head = make(); body0 = copy.deepcopy(body); init_head = copy.deepcopy(head.state_dict())
        opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
        msk = {"none": None, "all9": torch.ones(n + 1), "random9": torch.ones(n + 1), "others8": m_oth.float()}[kind]
        for step in range(0, max(Ks) + 1):
            if step > 0:
                i = rng.integers(0, len(tr), BS); z = head(body(Xtr[i])); tgt = Yrand[i] if kind == "random9" else Ytr[i]
                loss = (nn.functional.binary_cross_entropy_with_logits(z, tgt, reduction="none") * msk).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step in Ks: snaps.append((f"{kind}_K{step}", copy.deepcopy(body.state_dict()), body0, init_head))
    for tag, bsd, body0, init_head in snaps:
        body, _ = make(); body.load_state_dict(bsd); d = diag(body, body0, Xd)
        h2 = nn.Linear(128, n + 1); h2.load_state_dict(init_head); o2 = torch.optim.Adam(list(body.parameters()) + list(h2.parameters()), lr=1e-3)
        r2 = np.random.default_rng(seed * 100 + 77); first = None; prev_ok = False; prev_t = 0
        for step in range(1, CAP + 1):
            i = r2.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(body, h2, Xtr[i], True), Ytr[i], reduction="none").sum(1).mean(); o2.zero_grad(); loss.backward(); o2.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(body, h2, Xte, True)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        row = dict(seed=seed, cond=tag, cost=first if first is not None else -1, **{k: round(v, 4) for k, v in d.items()})
        if w is None:
            w = csv.DictWriter(fh, fieldnames=list(row)); 
            if os.path.getsize(fp) == 0: w.writeheader()
        w.writerow(row); fh.flush(); print(json.dumps(row), flush=True)
