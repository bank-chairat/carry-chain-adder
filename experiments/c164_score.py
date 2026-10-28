"""C-164 scoring: bit 5, training fraction 0.10, arms random and own5 (c164/)."""
"""C-157 part B / C-160 scoring (env BIT, default 6): bit-6 held-out error against bit 6's own dependency length d_6, own6 arm (c157/) against the C-08 random and M-stratified arms (c08/).
Pooled logistic regression (seed fixed effects) of bit-6 error on d_6; 95% interval by bootstrap over held-out examples within seed (500 draws)."""
import numpy as np, os
BIT = 5
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
    rng = np.random.default_rng(seed); allidx = rng.permutation(len(a)); te = allidx[:20000]; pool = allidx[20000:]; k = int(0.1 * len(a))
    s = {"random": rng.permutation(pool)[:k], "stratified": stratified(pool, k, rng)}; s[f"own{BIT}"] = own(pool, k, rng, BIT); return te, s
def own(pool, k, rng, bit=6):
    strata = {d: rng.permutation(pool[D[pool, bit] == d]) for d in range(bit + 1)}; quota = k // len(strata); chosen = [idx[:quota] for idx in strata.values()]
    chosen = np.concatenate(chosen); left = k - len(chosen); rest = np.setdiff1d(pool, chosen)
    return np.concatenate([chosen, rng.permutation(rest)[:left]])

def load(arm):
    return [(seed, np.load(f"c164/final_{seed}_{arm}.npz")["err"][:, BIT], np.load(f"c164/final_{seed}_{arm}.npz")["dep"][:, BIT]) for seed in range(30, 35)]
def coef(data):
    X = np.vstack([np.column_stack([d.astype(float)] + [np.full(len(d), float(s == t)) for t in range(31, 35)]) for s, e, d in data]); y = np.concatenate([e for s, e, d in data])
    return LogisticRegression(C=np.inf, max_iter=5000).fit(X, y).coef_[0][0]
rng = np.random.default_rng(0); res = {}; rates = {}
for arm in ["random", "own5"]:
    data = load(arm); c = coef(data); print(f"\n{arm} (fraction 0.10): bit-5 error by d_5 (pooled over seeds 30-34)")
    for dd in range(6):
        n_ = sum((d == dd).sum() for s, e, d in data); k_ = sum(e[d == dd].sum() for s, e, d in data)
        tc = sum((D[sets_for(s)[1][arm], 5] == dd).sum() for s, e, d in data); rates[(arm, dd)] = (k_, n_)
        print(f"  d_5={dd}: held-out {n_:6d} errors {int(k_):4d} rate {k_ / n_:.4f}  train {tc}")
    boots = []
    for b in range(500):
        bd = []
        for s, e, d in data: i = rng.integers(0, len(e), len(e)); bd.append((s, e[i], d[i]))
        boots.append(coef(bd))
    lo, hi = np.percentile(boots, [2.5, 97.5]); res[arm] = (c, lo, hi); print(f"  pooled d_5 coefficient {c:.3f} (95% bootstrap {lo:.3f} to {hi:.3f}); total errors {int(sum(e.sum() for s, e, d in data))}")
c, lo, hi = res["own5"]; print(f"\n(1) own5 coefficient negative, interval below 0: {'HIT' if hi < 0 else 'MISSED'} ({c:.3f}, {lo:.3f} to {hi:.3f})")
c, lo, hi = res["random"]; print(f"(2) random coefficient positive, interval above 0: {'HIT' if lo > 0 else 'MISSED'} ({c:.3f}, {lo:.3f} to {hi:.3f})")
(k4, n4), (k5, n5) = rates[("random", 4)], rates[("random", 5)]; p4, p5 = k4 / n4, k5 / n5; z = (p5 - p4) / np.hypot(np.sqrt(p4 * (1 - p4) / n4), np.sqrt(p5 * (1 - p5) / n5))
print(f"(3) random, d_5 = 4 vs 5 within 2 standard errors: {'HIT' if abs(z) <= 2 else 'MISSED'} ({p4:.4f} vs {p5:.4f}, {z:+.1f} SE)")
