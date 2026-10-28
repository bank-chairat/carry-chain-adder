"""C-07: dependency length vs carry run length (derived from adder_learn2.py, C-06). Same network/optimizer as adder_learn.py.
Usage: python3 adder_learn2.py OUTDIR [--smoke]   (resumable; writes OUTDIR/curves.csv and OUTDIR/bins.csv)
bins.csv: held-out error per (bit, carry-run bin) at every evaluation; bins 0, 1, 2, 3 (3 = three or more)."""
import sys, os, csv, time
import numpy as np, torch, torch.nn as nn
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv; os.makedirs(OUT, exist_ok=True)
NS, FRACS, SEEDS = ((4,), (0.5,), (10,)) if SMOKE else ((8,), (0.2, 0.05), range(10, 15))
STEPS = 300 if SMOKE else 10000; EVAL = 100; WIDTH, DEPTH, LR, BS = 128, 3, 1e-3, 256; MAXTEST = 50000
torch.set_num_threads(4)
def pairs(n):
    a, b = np.meshgrid(np.arange(2 ** n), np.arange(2 ** n), indexing="ij"); return a.ravel(), b.ravel()
def bits(x, n): return ((x[:, None] >> np.arange(n)) & 1).astype(np.float32)
def carries(a, b, n):
    c = np.zeros((len(a), n + 1), dtype=np.int64)
    for i in range(n):
        ai, bi = (a >> i) & 1, (b >> i) & 1; c[:, i + 1] = (ai & bi) | (c[:, i] & (ai ^ bi))
    return c
def runlen(c):
    r = np.zeros_like(c)
    for i in range(1, c.shape[1]): r[:, i] = np.where(c[:, i] == 1, r[:, i - 1] + 1, 0)
    return r
def deplen(a, b, n):
    p = ((a ^ b)[:, None] >> np.arange(n)) & 1; d = np.zeros((len(a), n + 1), dtype=np.int64)     # d[:, i] for output bit i
    for i in range(1, n + 1): d[:, i] = np.where(p[:, i - 1] == 1, d[:, i - 1] + 1, 0)
    return d
def net(d_in, d_out):
    L, d = [], d_in
    for _ in range(DEPTH): L += [nn.Linear(d, WIDTH), nn.ReLU()]; d = WIDTH
    L.append(nn.Linear(d, d_out)); return nn.Sequential(*L)

cp, bp = f"{OUT}/curves.csv", f"{OUT}/bins.csv"; done = set()
if os.path.exists(cp): done = {(r["n"], r["frac"], r["op"], r["seed"]) for r in csv.DictReader(open(cp))}
newc, newb = not os.path.exists(cp), not os.path.exists(bp)
fc, fb = open(cp, "a", newline=""), open(bp, "a", newline=""); wc, wb = csv.writer(fc), csv.writer(fb)
if newc: wc.writerow(["n", "frac", "op", "seed", "step", "bit", "test_acc"])
if newb: wb.writerow(["n", "frac", "op", "seed", "step", "bit", "bin", "n_examples", "test_error"])
for n in NS:
    a, b = pairs(n); X = np.concatenate([bits(a, n), bits(b, n)], 1); C = carries(a, b, n); R = np.minimum(runlen(C), 3); Dd = deplen(a, b, n)
    for frac in FRACS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed); perm = rng.permutation(len(a)); ntr = int(frac * len(a)); tr = perm[:ntr]; te = perm[ntr:][:MAXTEST]
            for op in ("add", "xor"):
                if (str(n), str(frac), op, str(seed)) in done: continue
                t0 = time.time(); Y = bits(a + b, n + 1) if op == "add" else bits(a ^ b, n)
                Xt, Yt = torch.tensor(X), torch.tensor(Y); torch.manual_seed(seed); m = net(2 * n, Y.shape[1])
                opt = torch.optim.Adam(m.parameters(), lr=LR); g = torch.Generator().manual_seed(1000 + seed); tri = torch.tensor(tr); tei = torch.tensor(te); Rt = R[te]
                for step in range(1, STEPS + 1):
                    ix = tri[torch.randint(0, len(tri), (BS,), generator=g)]
                    loss = nn.functional.binary_cross_entropy_with_logits(m(Xt[ix]), Yt[ix], reduction="none").sum(1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    if step % EVAL == 0:
                        with torch.no_grad(): err = (m(Xt[tei]) > 0).float().ne(Yt[tei]).numpy()
                        if step == STEPS: np.savez_compressed(f"{OUT}/final_{n}_{frac}_{op}_{seed}.npz", err=err, carry=C[te], run=runlen(C)[te], dep=Dd[te])
                        for k in range(Y.shape[1]):
                            wc.writerow([n, frac, op, seed, step, k, f"{1 - err[:, k].mean():.5f}"])
                            for bn in range(4):
                                msk = Rt[:, k] == bn
                                if msk.sum() > 0 and bn <= k: wb.writerow([n, frac, op, seed, step, k, bn, int(msk.sum()), f"{err[msk, k].mean():.5f}"])
                fc.flush(); fb.flush(); print(f"n={n} frac={frac} {op} seed={seed} final acc per bit {np.round(1 - err.mean(0), 3).tolist()} ({time.time() - t0:.0f}s)", flush=True)
print("done")
