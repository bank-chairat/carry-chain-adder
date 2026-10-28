"""C-157 part B: like adder_learn4.py (C-08: same test sets, pool, init and batch seeds), but the training set has equal quotas per bit 6's OWN dependency length d_6 = 0..6 (arm own6; env BIT selects another bit, arm ownBIT, used by C-160; env FRAC sets the training fraction and ARMS='own random' also trains the random set, used by C-164). Usage: [BIT=5] python3 c157_own6.py OUTDIR [--smoke]"""
import sys, os, csv, time
import numpy as np, torch, torch.nn as nn
BIT = int(os.environ.get('BIT', '6')); FRAC = float(os.environ.get('FRAC', '0.05')); ARMS = os.environ.get('ARMS', 'own').split(); OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv; os.makedirs(OUT, exist_ok=True)
n = 8; SEEDS = (30,) if SMOKE else range(30, 35); STEPS = 300 if SMOKE else 10000; WIDTH, DEPTH, LR, BS = 128, 3, 1e-3, 256; torch.set_num_threads(2)
a, b = np.meshgrid(np.arange(2 ** n), np.arange(2 ** n), indexing="ij"); a, b = a.ravel(), b.ravel()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
X = np.concatenate([bits(a, n), bits(b, n)], 1); Y = bits(a + b, n + 1)
p = ((a ^ b)[:, None] >> np.arange(n)) & 1; D = np.zeros((len(a), n + 1), dtype=np.int64)
for i in range(1, n + 1): D[:, i] = np.where(p[:, i - 1] == 1, D[:, i - 1] + 1, 0)
M = D.max(1)
def net():
    L, d = [], 2 * n
    for _ in range(DEPTH): L += [nn.Linear(d, WIDTH), nn.ReLU()]; d = WIDTH
    L.append(nn.Linear(d, n + 1)); return nn.Sequential(*L)
def own(pool, k, rng, bit=6):
    strata = {d: rng.permutation(pool[D[pool, bit] == d]) for d in range(bit + 1)}; quota = k // len(strata); chosen = [idx[:quota] for idx in strata.values()]
    assert all(len(c) == quota for c in chosen); chosen = np.concatenate(chosen); left = k - len(chosen); rest = np.setdiff1d(pool, chosen)
    return np.concatenate([chosen, rng.permutation(rest)[:left]])
def stratified(pool, k, rng):
    strata = {m: rng.permutation(pool[M[pool] == m]) for m in np.unique(M[pool])}; chosen = []; quota = k // len(strata); left = k
    for m, idx in strata.items(): take = idx[:min(quota, len(idx))]; chosen.append(take); left -= len(take)
    chosen = np.concatenate(chosen); rest = np.setdiff1d(pool, chosen); extra = rng.permutation(rest)[:left]
    return np.concatenate([chosen, extra])
sp = f"{OUT}/summary.csv"; new = not os.path.exists(sp); f = open(sp, "a", newline=""); w = csv.writer(f)
if new: w.writerow(["seed", "scheme", "n_train", "test_acc_all_bits_mean", "test_acc_min_bit"])
done = {(r["seed"], r["scheme"]) for r in csv.DictReader(open(sp))} if not new else set()
for seed in SEEDS:
    rng = np.random.default_rng(seed); allidx = rng.permutation(len(a)); te = allidx[:20000]; pool = allidx[20000:]; k = int(FRAC * len(a))
    sets = {"random": rng.permutation(pool)[:k], "stratified": stratified(pool, k, rng)}
    sets = {kk: v for kk, v in {f"own{BIT}": own(pool, k, rng, BIT), "random": sets["random"]}.items() if kk.rstrip("0123456789") in ARMS}  # C-08 sets drawn first so the RNG state matches  # the two C-08 sets are drawn first so the RNG state matches; only own6 is trained
    for scheme, tr in sets.items():
        if (str(seed), scheme) in done: continue
        t0 = time.time(); Xt, Yt = torch.tensor(X), torch.tensor(Y); torch.manual_seed(seed); m = net(); opt = torch.optim.Adam(m.parameters(), lr=LR)
        g = torch.Generator().manual_seed(1000 + seed); tri = torch.tensor(tr); tei = torch.tensor(te)
        for step in range(1, STEPS + 1):
            ix = tri[torch.randint(0, len(tri), (BS,), generator=g)]
            loss = nn.functional.binary_cross_entropy_with_logits(m(Xt[ix]), Yt[ix], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): err = (m(Xt[tei]) > 0).float().ne(Yt[tei]).numpy()
        np.savez_compressed(f"{OUT}/final_{seed}_{scheme}.npz", err=err, dep=D[te], M_train=M[tr])
        w.writerow([seed, scheme, len(tr), f"{1 - err.mean():.5f}", f"{(1 - err.mean(0)).min():.5f}"]); f.flush()
        print(f"seed={seed} {scheme} acc={1 - err.mean():.4f} min bit {(1 - err.mean(0)).min():.4f} ({time.time() - t0:.0f}s)", flush=True)
print("done")
