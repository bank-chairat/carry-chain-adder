"""C-01 and C-05: learning order of output bits and held-out error against carry run length, binary addition versus XOR, plain MLP.
Usage: python3 adder_learn.py OUTDIR [--smoke]     (resumable; writes OUTDIR/curves.csv, OUTDIR/by_runlen.csv)
Design is fixed in PROGRESS.md (section 'Design fixed before the first run')."""
import sys, os, csv, time
import numpy as np, torch, torch.nn as nn
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv; os.makedirs(OUT, exist_ok=True)
NS, SEEDS = ((4,), (0,)) if SMOKE else ((6, 8), range(5))
STEPS = 300 if SMOKE else 10000; EVAL = 100 if SMOKE else 100; TAB = (100, 200, 300) if SMOKE else (2000, 5000, 10000)
WIDTH, DEPTH, LR, BS = 128, 3, 1e-3, 256
torch.set_num_threads(4)

def pairs(n):
    a, b = np.meshgrid(np.arange(2 ** n), np.arange(2 ** n), indexing="ij"); return a.ravel(), b.ravel()
def bits(x, n): return ((x[:, None] >> np.arange(n)) & 1).astype(np.float32)      # least significant first
def carries(a, b, n):
    c = np.zeros((len(a), n + 1), dtype=np.int64)                                  # c[:, i] = carry into bit i, c[:, n] = carry out
    for i in range(n):
        ai, bi = (a >> i) & 1, (b >> i) & 1; c[:, i + 1] = (ai & bi) | (c[:, i] & (ai ^ bi))
    return c
def runlen(c):
    r = np.zeros_like(c)
    for i in range(1, c.shape[1]): r[:, i] = np.where(c[:, i] == 1, r[:, i - 1] + 1, 0)
    return r                                                                       # number of consecutive ones in c_i, c_{i-1}, ...
def net(d_in, d_out):
    L, d = [], d_in
    for _ in range(DEPTH): L += [nn.Linear(d, WIDTH), nn.ReLU()]; d = WIDTH
    L.append(nn.Linear(d, d_out)); return nn.Sequential(*L)

cp, tp = f"{OUT}/curves.csv", f"{OUT}/by_runlen.csv"
done = set()
if os.path.exists(cp): done = {(r["n"], r["op"], r["seed"]) for r in csv.DictReader(open(cp))}
newc, newt = not os.path.exists(cp), not os.path.exists(tp)
fc, ft = open(cp, "a", newline=""), open(tp, "a", newline=""); wc, wt = csv.writer(fc), csv.writer(ft)
if newc: wc.writerow(["n", "op", "seed", "step", "bit", "test_acc", "train_acc"])
if newt: wt.writerow(["n", "op", "seed", "step", "bit", "runlen", "n_examples", "test_error"])
for n in NS:
    a, b = pairs(n); X = np.concatenate([bits(a, n), bits(b, n)], 1); c = carries(a, b, n); R = runlen(c)
    for seed in SEEDS:
        rng = np.random.default_rng(seed); perm = rng.permutation(len(a)); tr, te = perm[: len(a) // 2], perm[len(a) // 2:]
        for op in ("add", "xor"):
            if (str(n), op, str(seed)) in done: continue
            t0 = time.time()
            Y = bits(a + b, n + 1) if op == "add" else bits(a ^ b, n)
            Xt, Yt = torch.tensor(X), torch.tensor(Y); torch.manual_seed(seed); m = net(2 * n, Y.shape[1])
            opt = torch.optim.Adam(m.parameters(), lr=LR); g = torch.Generator().manual_seed(1000 + seed); tri = torch.tensor(tr); tei = torch.tensor(te)
            for step in range(1, STEPS + 1):
                ix = tri[torch.randint(0, len(tri), (BS,), generator=g)]
                loss = nn.functional.binary_cross_entropy_with_logits(m(Xt[ix]), Yt[ix], reduction="none").sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if step % EVAL == 0 or step in TAB:
                    with torch.no_grad():
                        pt = (m(Xt[tei]) > 0).float(); ptr = (m(Xt[tri]) > 0).float()
                        acc_te = (pt == Yt[tei]).float().mean(0).numpy(); acc_tr = (ptr == Yt[tri]).float().mean(0).numpy()
                    if step % EVAL == 0:
                        for k in range(Y.shape[1]): wc.writerow([n, op, seed, step, k, f"{acc_te[k]:.5f}", f"{acc_tr[k]:.5f}"])
                    if step in TAB:
                        err = (pt != Yt[tei]).numpy(); Rt = R[te]
                        for k in range(Y.shape[1]):
                            for L in range(0, k + 1):
                                msk = Rt[:, k] == L
                                if msk.sum() > 0: wt.writerow([n, op, seed, step, k, L, int(msk.sum()), f"{err[msk, k].mean():.5f}"])
            fc.flush(); ft.flush(); print(f"n={n} {op} seed={seed} final held-out acc per bit {np.round(acc_te, 3).tolist()} ({time.time() - t0:.0f}s)", flush=True)
print("done")
