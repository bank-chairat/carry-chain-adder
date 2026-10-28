import csv, copy, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8; BS = 256; K = 250
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); SUM = (A + B).astype(np.float64)
def make():
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L), nn.Linear(d, n + 1)
def r2_on(h, s):
    hc = h - h.mean(0); sc = (s - s.mean()) / s.std(); beta = (hc * sc[:, None]).mean(0); return float((beta ** 2).sum() / hc.var(0).sum())
rows = []
for seed in range(5):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr = perm[:len(A) // 2]
    Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr]); sel = np.random.default_rng(7).choice(len(tr), 8192, replace=False); Xd = Xtr[sel]; Sd = SUM[tr][sel]
    sub = np.random.default_rng(900 + seed); masks = np.zeros((9, 16), np.float32)
    for j in range(9): masks[j, sub.choice(16, 4, replace=False)] = 1
    Yrand = torch.tensor((X @ masks.T) % 2)[tr]
    for kind in ("init", "all9", "others8", "random9", "upper4", "lower4"):
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4); body, head = make()
        opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
        msk = torch.zeros(9)
        if kind in ("all9", "random9"): msk[:] = 1
        if kind == "others8": msk[:] = 1; msk[4] = 0
        if kind == "upper4": msk[5:] = 1
        if kind == "lower4": msk[:4] = 1
        if kind != "init":
            for step in range(K):
                i = rng.integers(0, len(tr), BS); z = head(body(Xtr[i])); tgt = Yrand[i] if kind == "random9" else Ytr[i]
                loss = (nn.functional.binary_cross_entropy_with_logits(z, tgt, reduction="none") * msk).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            h = Xd; out = {}
            for li, idx in enumerate((0, 2, 4)):
                h = body[idx + 1](body[idx](h)); hn = h.numpy().astype(np.float64); hc = hn - hn.mean(0)
                u, s, vt = np.linalg.svd(hc, full_matrices=False); pc1 = hc @ vt[0]
                out[f"L{li+1}_pc1corr"] = abs(np.corrcoef(pc1, Sd)[0, 1]); out[f"L{li+1}_R2S"] = r2_on(hn, Sd); out[f"L{li+1}_pc1share"] = float(s[0] ** 2 / (s ** 2).sum())
        rows.append(dict(seed=seed, kind=kind, **{k: round(v, 4) for k, v in out.items()})); print(rows[-1], flush=True)
with open("c113a.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for kind in ("init", "all9", "others8", "random9", "upper4", "lower4"):
    rr = [r for r in rows if r["kind"] == kind]; med = lambda k: np.median([r[k] for r in rr])
    print(kind.ljust(8), " ".join(f"{k}={med(k):.3f}" for k in ("L2_pc1corr", "L2_R2S", "L2_pc1share", "L3_pc1corr", "L3_R2S", "L3_pc1share")))
