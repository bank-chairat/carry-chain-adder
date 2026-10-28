import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 12; EV = 100; BS = 256; STEPS = 20000; CONDS = ["none", "p4", "c4", "p4c4"]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
_r = np.random.default_rng(12345); _ids = np.unique(_r.integers(0, 1 << (2 * n), 330000))[:320000]; _r.shuffle(_ids)
A = _ids >> n; B = _ids & ((1 << n) - 1)
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
p4 = (((A ^ B) >> 4) & 1).astype(np.float32)[:, None]; c4 = ((((A & 15) + (B & 15)) >> 4) & 1).astype(np.float32)[:, None]
assert (((p4[:, 0].astype(int) ^ c4[:, 0].astype(int)) == S[:, 4].astype(int)).all())
tr, te = np.arange(300000), np.arange(300000, 320000)
fp = f"{OUT}/aux12_n12.csv"; done = set()
if os.path.exists(fp): done = {(r["cond"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["cond", "seed", "time_bit4", "acc_bit4"])
for seed in range(5):
    for cond in CONDS:
        if (cond, str(seed)) in done: continue
        extra = {"none": np.zeros((len(A), 0), np.float32), "p4": p4, "c4": c4, "p4c4": np.concatenate([p4, c4], 1)}[cond]
        Y = np.concatenate([S, extra], 1)
        Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        L, d = [], 2 * n
        for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        net = nn.Sequential(*L, nn.Linear(d, Y.shape[1])); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        first = -1; prev_ok = False; prev_t = 0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if first < 0 and prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([cond, seed, first, acc]); fh.flush(); print(cond, seed, first, round(acc, 4), flush=True)
