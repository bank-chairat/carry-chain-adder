"""C-12: each sum bit trained alone (single-output net), n=8, random 50% of pairs for training, held-out accuracy every 100 steps.
Usage: SEEDS="0 1 2" python3 c12_single.py OUTDIR  (one thread, resumable per (bit, seed))"""
import sys, os, csv, time, json
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
n = 12; STEPS = int(os.environ.get('STEPS','20000')); _unused = 0;  EV = 100; BS = 256
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
_r = np.random.default_rng(12345); _ids = np.unique(_r.integers(0, 1 << (2 * n), 330000))[:320000]; _r.shuffle(_ids)
A = _ids >> n; B = _ids & ((1 << n) - 1)
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
fp = f"{OUT}/single_n12.csv"; done = set()
if os.path.exists(fp): done = {(r["bit"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["bit", "seed", "time_to_099", "final_acc"])
for seed in SEEDS:
    prng = np.random.default_rng(500 + seed); tr, te = np.arange(300000), np.arange(300000, 320000)
    Xtr, Xte = torch.tensor(X[tr]), torch.tensor(X[te])
    for bit in range(n + 1):
        if (str(bit), str(seed)) in done: continue
        torch.manual_seed(seed * 100 + bit); rng = np.random.default_rng(seed * 100 + bit)
        L, d = [], 2 * n
        for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        net = nn.Sequential(*L, nn.Linear(d, 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        ytr, yte = torch.tensor(Y[tr, bit:bit + 1]), torch.tensor(Y[te, bit:bit + 1])
        first = -1; prev_ok = False; prev_t = 0; t0 = time.time()
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i]); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte) > 0).float() == yte).float().mean().item()
                good = acc >= 0.99
                if first < 0 and prev_ok and good: first = prev_t
                prev_ok = good; prev_t = step
                if first >= 0: break
        w.writerow([bit, seed, first, acc]); fh.flush()
        print(f"bit={bit} seed={seed} t={first} acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
