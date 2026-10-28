"""C-157 design info: per-bit own dependency-length counts in the C-08 training sets (rebuilt with the same RNG calls as adder_learn4.py). No outcomes are read."""
import numpy as np
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
for seed in [30]:
    rng = np.random.default_rng(seed); allidx = rng.permutation(len(a)); te = allidx[:20000]; pool = allidx[20000:]; k = int(0.05 * len(a))
    sets = {"random": rng.permutation(pool)[:k], "stratified": stratified(pool, k, rng)}
    for s, tr in sets.items():
        print(s, "M counts", np.bincount(M[tr], minlength=9))
        for i in range(4, 9): print(f"  bit {i} own d counts", np.bincount(D[tr, i], minlength=i + 1))
    print("pool size", len(pool)); 
    for i in [5, 6, 7]: print(f"pool own d_{i} counts", np.bincount(D[pool, i], minlength=i + 1))
