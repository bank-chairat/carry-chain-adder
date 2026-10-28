"""C-157 part B / C-160 scoring (env BIT, default 6): bit-6 held-out error against bit 6's own dependency length d_6, own6 arm (c157/) against the C-08 random and M-stratified arms (c08/).
Pooled logistic regression (seed fixed effects) of bit-6 error on d_6; 95% interval by bootstrap over held-out examples within seed (500 draws)."""
import numpy as np, os
BIT = int(os.environ.get('BIT', '6'))
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
    s = {"random": rng.permutation(pool)[:k], "stratified": stratified(pool, k, rng)}; s[f"own{BIT}"] = own(pool, k, rng, BIT); return te, s
def own(pool, k, rng, bit=6):
    strata = {d: rng.permutation(pool[D[pool, bit] == d]) for d in range(bit + 1)}; quota = k // len(strata); chosen = [idx[:quota] for idx in strata.values()]
    chosen = np.concatenate(chosen); left = k - len(chosen); rest = np.setdiff1d(pool, chosen)
    return np.concatenate([chosen, rng.permutation(rest)[:left]])
def load(arm):
    out = []
    for seed in range(30, 35):
        z = np.load(f"{'c157' if BIT == 6 else 'c160'}/final_{seed}_own{BIT}.npz") if arm.startswith("own") else np.load(f"c08/final_{seed}_{arm}.npz")
        out.append((seed, z["err"][:, BIT], z["dep"][:, BIT]))
    return out
def coef(data):
    X = np.vstack([np.column_stack([d.astype(float)] + [np.full(len(d), float(s == t)) for t in range(31, 35)]) for s, e, d in data]); y = np.concatenate([e for s, e, d in data])
    return LogisticRegression(C=np.inf, max_iter=5000).fit(X, y).coef_[0][0]
rng = np.random.default_rng(0); res = {}
for arm in ["random", "stratified", f"own{BIT}"]:
    data = load(arm); c = coef(data)
    print(f"\n{arm}: bit-{BIT} error by d_{BIT} (pooled over seeds 30-34; training count per d summed over seeds)")
    for dd in range(BIT + 1):
        n_ = sum((d == dd).sum() for s, e, d in data); k_ = sum(e[d == dd].sum() for s, e, d in data)
        tc = 0
        for s, e, d in data:
            te, st = sets_for(s)
            tc += (D[st[arm], BIT] == dd).sum()
        print(f"  d_{BIT}={dd}: held-out {n_:6d} errors {int(k_):4d} rate {k_ / n_:.4f}" + f"  train {tc}")
    boots = []
    for b in range(500):
        bd = []
        for s, e, d in data: i = rng.integers(0, len(e), len(e)); bd.append((s, e[i], d[i]))
        boots.append(coef(bd))
    lo, hi = np.percentile(boots, [2.5, 97.5]); res[arm] = (c, lo, hi)
    print(f"  pooled d_{BIT} coefficient {c:.3f} (95% bootstrap {lo:.3f} to {hi:.3f}); total bit-{BIT} errors {int(sum(e.sum() for s, e, d in data))}")
c, lo, hi = res[f"own{BIT}"]
if BIT == 6: print(f"\n(B1) own6 coefficient positive with interval excluding 0: {'HIT' if lo > 0 else 'MISSED'} ({c:.3f}, {lo:.3f} to {hi:.3f})")
if BIT == 6: print(f"(B2) own6 coefficient at least half of the stratified one ({res['stratified'][0]:.3f}): {'HIT' if c >= 0.5 * res['stratified'][0] else 'MISSED'} (ratio {c / res['stratified'][0]:.2f})")
if BIT != 6:
    data = load(f"own{BIT}"); r = lambda dd: sum(e[d == dd].sum() for s_, e, d in data) / sum((d == dd).sum() for s_, e, d in data)
    print(f"\nC-160 (1) own{BIT} coefficient negative with interval below 0: {'HIT' if hi < 0 else 'MISSED'} ({c:.3f}, {lo:.3f} to {hi:.3f})")
    print(f"C-160 (2) error at d = {BIT} below error at d = 0: {'HIT' if r(BIT) < r(0) else 'MISSED'} ({r(BIT):.4f} vs {r(0):.4f})")
