"""C-157 part A: per (bit, own d) errors in the C-08 networks and a regression with the training count of each cell.
First reproduces the C-08 coefficients (bits 4-8, bit effects, d) to check the regression definition."""
import numpy as np
from sklearn.linear_model import LogisticRegression
n = 8
a, b = np.meshgrid(np.arange(2 ** n), np.arange(2 ** n), indexing="ij"); a, b = a.ravel(), b.ravel()
p = ((a ^ b)[:, None] >> np.arange(n)) & 1; D = np.zeros((len(a), n + 1), dtype=np.int64)
for i in range(1, n + 1): D[:, i] = np.where(p[:, i - 1] == 1, D[:, i - 1] + 1, 0)
M = D.max(1)
def stratified(pool, k, rng):
    strata = {m: rng.permutation(pool[M[pool] == m]) for m in np.unique(M[pool])}; chosen = []; quota = k // len(strata); left = k
    for m, idx in strata.items(): take = idx[:min(quota, len(idx))]; chosen.append(take); left -= len(take)
    chosen = np.concatenate(chosen); rest = np.setdiff1d(pool, chosen); extra = rng.permutation(rest)[:left]
    return np.concatenate([chosen, extra])
def sets_for(seed):
    rng = np.random.default_rng(seed); allidx = rng.permutation(len(a)); te = allidx[:20000]; pool = allidx[20000:]; k = int(0.05 * len(a))
    return te, {"random": rng.permutation(pool)[:k], "stratified": stratified(pool, k, rng)}
def fit(feats, y):
    clf = LogisticRegression(C=np.inf, max_iter=5000).fit(feats, y); return clf.coef_[0]
def design(err, dep, bitsel, extra=None):
    rows, ys = [], []
    for i in bitsel:
        f = [dep[:, i][:, None]] + [np.full((len(dep), 1), float(i == j)) for j in bitsel[1:]]
        if extra is not None: f.append(extra[i][:, None])
        rows.append(np.hstack(f)); ys.append(err[:, i])
    return np.vstack(rows).astype(float), np.concatenate(ys)
print("Check: C-08 coefficient of d (bits 4-8, bit effects)")
for scheme in ["random", "stratified"]:
    cs = []
    for seed in range(30, 35):
        z = np.load(f"c08/final_{seed}_{scheme}.npz"); X_, y_ = design(z["err"], z["dep"], [4, 5, 6, 7, 8]); cs.append(fit(X_, y_)[0])
    print(scheme, " ".join(f"{c:.2f}" for c in cs))
print("\nPart A: stratified arm, pooled over seeds 30-34: per bit, own d: training count (summed over seeds), held-out n, errors, error rate")
cells = {}
for seed in range(30, 35):
    te, sets = sets_for(seed); tr = sets["stratified"]; z = np.load(f"c08/final_{seed}_stratified.npz"); err, dep = z["err"], z["dep"]
    assert (dep == D[te]).all()
    for i in range(5, 9):
        cnt = np.bincount(D[tr, i], minlength=i + 1)
        for d in range(i + 1):
            m = dep[:, i] == d; c = cells.setdefault((i, d), [0, 0, 0]); c[0] += cnt[d]; c[1] += m.sum(); c[2] += err[m, i].sum()
res = {}
for i in range(5, 9):
    print(f"bit {i}")
    for d in range(i + 1):
        c = cells[(i, d)]; print(f"  d={d}: train {c[0]:5d}  heldout {c[1]:6d}  errors {c[2]:4d}  rate {c[2] / max(c[1], 1):.4f}")
    if i >= 6:
        rarest = min(range(i + 1), key=lambda d: cells[(i, d)][0]); r_long = cells[(i, i)][2] / cells[(i, i)][1]; r_rare = cells[(i, rarest)][2] / cells[(i, rarest)][1]
        res[i] = r_long > r_rare; print(f"  A1 bit {i}: rarest-count d = {rarest} rate {r_rare:.4f}; longest d = {i} rate {r_long:.4f} -> longest higher: {r_long > r_rare}")
print("A1 hits:", sum(res.values()), "of 3 (need 2)")
print("\nA2: per seed, coefficient of d with bit effects (bits 5-8) and log training count of the (bit, d) cell")
pos = 0
for seed in range(30, 35):
    te, sets = sets_for(seed); tr = sets["stratified"]; z = np.load(f"c08/final_{seed}_stratified.npz"); err, dep = z["err"], z["dep"]
    lc = {i: np.log(np.bincount(D[tr, i], minlength=i + 1)[dep[:, i]] + 1.0) for i in range(5, 9)}
    X0, y0 = design(err, dep, [5, 6, 7, 8]); X1, y1 = design(err, dep, [5, 6, 7, 8], lc)
    c0 = fit(X0, y0); c1 = fit(X1, y1); pos += c1[0] > 0
    print(f"seed {seed}: d coef without count {c0[0]:.3f}; with log count: d {c1[0]:.3f}, log count {c1[-1]:.3f}; errors {int(y0.sum())}")
print("A2 positive in", pos, "of 5 (need 4)")
