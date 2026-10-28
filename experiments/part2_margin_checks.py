"""Checks R-1, V-1, R-5 (C-13), V-4, R-8 and R-9 of PROGRESS.md, recomputed from scratch.
R-1 and V-1 are exhaustive over all input pairs (no training). R-5, V-4 and R-9 use the saved held-out
errors of the C-13 networks (c13a/, c13b/, written by adder_learn5.py). R-8 simulates two idealised adders.
(The original one-off analysis code for these checks was not kept; this script re-implements them from the
definitions in PROGRESS.md, and its output is compared there.) Run: python3 part2_margin_checks.py"""
import glob
import numpy as np
from scipy.stats import spearmanr

n = 8
a, b = np.meshgrid(np.arange(2 ** n), np.arange(2 ** n), indexing="ij"); a, b = a.ravel(), b.ravel()
p = ((a ^ b)[:, None] >> np.arange(n)) & 1
D = np.zeros((len(a), n + 1), dtype=np.int64)              # dependency length d_i (as in adder_learn5.py)
for i in range(1, n + 1): D[:, i] = np.where(p[:, i - 1] == 1, D[:, i - 1] + 1, 0)
def S(i): return (a & (2 ** i - 1)) + (b & (2 ** i - 1))   # weighted sum of the low i bits
def margin(i): return S(i) - (2 ** i - 0.5)                 # distance from the carry threshold

print("R-1  'd_i >= k implies |m_i| < 2^(i-k)' on all 65,536 pairs")
viol = sum(int(np.sum((D[:, i] >= k) & (np.abs(margin(i)) >= 2.0 ** (i - k)))) for i in range(2, n + 1) for k in range(1, i + 1))
print(f"  violations: {viol}")
print("  Spearman(d_i, -log2(|m_i|+1)), i = 2..8: " + " ".join(f"{spearmanr(D[:, i], -np.log2(np.abs(margin(i)) + 1)).statistic:.3f}" for i in range(2, n + 1)))

print("\nV-1  one-hidden-layer ReLU adder with 3n - 1 units, all pairs")
relu = lambda z: np.maximum(z, 0)
for m in (3, 4, 5, 6, 8, 10):
    A, B = np.meshgrid(np.arange(2 ** m), np.arange(2 ** m), indexing="ij"); A, B = A.ravel(), B.ravel()
    ab = lambda j: ((A >> j) & 1) + ((B >> j) & 1)
    Si = lambda i: sum((2 ** j) * ab(j) for j in range(i))
    tt = [relu(ab(j)) for j in range(m)]                                  # m units
    c = {1: relu(Si(1) - 1)}                                              # 1 unit
    for i in range(2, m + 1): c[i] = relu(Si(i) - 2 ** i + 1) - relu(Si(i) - 2 ** i)   # 2 units each
    y = [tt[i] + (c[i] if i >= 1 else 0) - 2 * c[i + 1] for i in range(m)] + [c[m]]
    want = [((A + B) >> i) & 1 for i in range(m + 1)]
    units = m + 1 + 2 * (m - 1)
    print(f"  n = {m}: {units} units, mismatches {sum(int(np.sum(np.abs(yi - wi) > 1e-9)) for yi, wi in zip(y, want))}")

runs = sorted(glob.glob("c13a/final_*_random.npz") + glob.glob("c13b/final_*_random.npz"), key=lambda f: int(f.split("_")[-2]))
print("\nR-5  error on d_i = 0 examples near (SMALL) and far from (LARGE) the threshold, bits 5-8 pooled")
v4 = []
for f in runs:
    z = np.load(f); te, err = z["te"], z["err"]; seed = int(f.split("_")[-2])
    es, ns, el, nl = 0, 0, 0, 0
    for i in range(5, n + 1):
        m_, d0 = np.abs(margin(i)[te]), D[te, i] == 0
        sm, lg = d0 & (m_ <= 2 ** (i - 2)), d0 & (m_ > 2 ** (i - 1))
        es += err[sm, i].sum(); ns += sm.sum(); el += err[lg, i].sum(); nl += lg.sum()
    small, large = es / ns, el / nl
    # V-4: d_i >= 3 against d_i = 0 at equal margin, factor-2 bins, bins with at least 50 in both groups
    e3 = n3 = e0 = n0 = 0
    for i in range(5, n + 1):
        m_, d = np.abs(margin(i)[te]), D[te, i]; edges = [0, 1] + [2.0 ** k for k in range(1, i + 1)]
        for lo, hi in zip(edges, edges[1:]):
            inb = (m_ > lo) & (m_ <= hi); g3, g0 = inb & (d >= 3), inb & (d == 0)
            if g3.sum() >= 50 and g0.sum() >= 50: e3 += err[g3, i].sum(); n3 += g3.sum(); e0 += err[g0, i].sum(); n0 += g0.sum()
    v4.append((e3 / n3) / (e0 / n0))
    print(f"  seed {seed}: SMALL {small:.4f}  LARGE {large:.4f}  ratio {small / max(large, 1e-12):.1f}  | V-4 ratio {v4[-1]:.2f}")
print(f"  V-4 median ratio {np.median(v4):.2f}")

print("\nR-8  the same two metrics for two idealised adders on all pairs (bits 5-8; overall error over all 9 bits)")
true = [((a + b) >> i) & 1 for i in range(n + 1)]
def metrics(pred):
    err = np.stack([pred[i] != true[i] for i in range(n + 1)], 1)
    es = ns = el = nl = e3 = n3 = e0 = n0 = 0
    for i in range(5, n + 1):
        m_, d = np.abs(margin(i)), D[:, i]; sm, lg = (d == 0) & (m_ <= 2 ** (i - 2)), (d == 0) & (m_ > 2 ** (i - 1))
        es += err[sm, i].sum(); ns += sm.sum(); el += err[lg, i].sum(); nl += lg.sum()
        edges = [0, 1] + [2.0 ** k for k in range(1, i + 1)]
        for lo, hi in zip(edges, edges[1:]):
            inb = (m_ > lo) & (m_ <= hi); g3, g0 = inb & (d >= 3), inb & (d == 0)
            if g3.sum() >= 50 and g0.sum() >= 50: e3 += err[g3, i].sum(); n3 += g3.sum(); e0 += err[g0, i].sum(); n0 += g0.sum()
    v = (e3 / n3) / (e0 / n0) if e0 > 0 else float("inf")
    return err.mean(), es / ns, el / nl, v
rng = np.random.default_rng(0)
for sig in (0.5, 1, 2, 4):     # M1: thresholded weighted sum with Gaussian noise
    ch = {i: (S(i) + rng.normal(0, sig, len(a)) >= 2 ** i - 0.5).astype(int) for i in range(1, n + 1)}
    pred = [p[:, 0]] + [p[:, i] ^ ch[i] for i in range(1, n)] + [ch[n]]
    print("  M1 noisy threshold, sigma %g: overall %.3f  SMALL %.4f  LARGE %.4f  V-4 ratio %.1f" % ((sig,) + metrics(pred)))
gk = ((a & b)[:, None] >> np.arange(n)) & 1
for K in (2, 3, 4):            # M2: ripple carry computed from the top K positions only
    def carry(i):
        c = np.zeros(len(a), dtype=int)
        for j in range(max(0, i - K), i): c = gk[:, j] | (p[:, j] & c)
        return c
    pred = [p[:, 0]] + [p[:, i] ^ carry(i) for i in range(1, n)] + [carry(n)]
    print("  M2 truncated ripple, K = %d: overall %.3f  SMALL %.4f  LARGE %.4f  V-4 ratio %s" % ((K,) + metrics(pred)[:3] + (f"{metrics(pred)[3]:.1f}",)))

print("\nR-9  does the distance to the nearest training pair explain the SMALL excess? (logistic regression)")
from sklearn.linear_model import LogisticRegression
Xbits = np.concatenate([((a[:, None] >> np.arange(n)) & 1), ((b[:, None] >> np.arange(n)) & 1)], 1).astype(np.int8)
for f in runs:
    seed = int(f.split("_")[-2]); z = np.load(f); te, err = z["te"], z["err"]
    rng = np.random.default_rng(seed); allidx = rng.permutation(len(a)); pool = allidx[20000:]; tr = rng.permutation(pool)[:3276]
    assert np.array_equal(allidx[:20000], te)
    dist = np.min(np.count_nonzero(Xbits[te][:, None, :] != Xbits[tr][None, :, :], axis=2), axis=1)
    feats, ys = [], []
    for i in range(5, n + 1):
        m_, d0 = np.abs(margin(i)[te]), D[te, i] == 0
        keep = d0 & ((m_ <= 2 ** (i - 2)) | (m_ > 2 ** (i - 1)))
        onehot = np.zeros((keep.sum(), 4)); onehot[:, i - 5] = 1
        feats.append(np.column_stack([onehot, (m_[keep] <= 2 ** (i - 2)).astype(float), dist[keep]])); ys.append(err[keep, i])
    F, y = np.concatenate(feats), np.concatenate(ys)
    fit = lambda cols: LogisticRegression(C=1e6, max_iter=5000, fit_intercept=False).fit(F[:, cols], y).coef_[0][4]
    print(f"  seed {seed}: SMALL coefficient {fit([0, 1, 2, 3, 4]):.2f} -> {fit([0, 1, 2, 3, 4, 5]):.2f} with distance; corr(SMALL, distance) {np.corrcoef(F[:, 4], F[:, 5])[0, 1]:+.2f}")
