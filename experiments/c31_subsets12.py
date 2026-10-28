"""C-15: time-to-0.99 of bit 4 (n=8) when trained jointly with different output subsets. Usage: python3 c15_subsets.py OUTDIR (one thread, resumable)"""
import sys, os, csv, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 12; EV = 100; BS = 256; STEPS = 20000; TARGET = 4
import json
SETS = json.loads(os.environ["SETS"]) if "SETS" in os.environ else [[4], [4, 0], [4, 8], [3, 4, 5], [2, 3, 4, 5, 6], list(range(9))]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
_r = np.random.default_rng(12345); _ids = np.unique(_r.integers(0, 1 << (2 * n), 330000))[:320000]; _r.shuffle(_ids)
A = _ids >> n; B = _ids & ((1 << n) - 1)
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
fp = f"{OUT}/subsets_n12.csv"; done = set()
if os.path.exists(fp): done = {(r["set"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["set", "seed", "time_bit4", "acc_bit4"])
for seed in range(5):
    tr, te = np.arange(300000), np.arange(300000, 320000)
    Xtr, Xte = torch.tensor(X[tr]), torch.tensor(X[te])
    for S in SETS:
        key = "-".join(map(str, S))
        if (key, str(seed)) in done: continue
        torch.manual_seed(seed * 100 + TARGET); rng = np.random.default_rng(seed * 100 + TARGET)
        L, d = [], 2 * n
        for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        net = nn.Sequential(*L, nn.Linear(d, len(S))); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        ytr, yte = torch.tensor(Y[tr][:, S]), torch.tensor(Y[te][:, S]); col = S.index(TARGET)
        first = -1; prev_ok = False; prev_t = 0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte)[:, col] > 0).float() == yte[:, col]).float().mean().item()
                good = acc >= 0.99
                if first < 0 and prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([key, seed, first, acc]); fh.flush(); print(key, seed, first, round(acc, 4), flush=True)
