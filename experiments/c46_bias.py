import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 100; BS = 256; STEPS = 10000; QS = [0.5, 0.35, 0.2]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
fp = f"{OUT}/bias_n8.csv"; done = set()
if os.path.exists(fp): done = {(r["q"], r["bit"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["q", "bit", "seed", "time_to_099", "final_acc", "trivial_acc"])
for q in QS:
    for seed in range(3):
        r = np.random.default_rng(300 + seed)
        ab = (r.random((120000, 2 * n)) < q).astype(np.int64); A = (ab[:, :n] << np.arange(n)).sum(1); B = (ab[:, n:] << np.arange(n)).sum(1)
        X = ab.astype(np.float32); S = bits(A + B, n + 1); tr, te = np.arange(100000), np.arange(100000, 120000)
        Xtr, Xte = torch.tensor(X[tr]), torch.tensor(X[te])
        for bit in range(n + 1):
            if (str(q), str(bit), str(seed)) in done: continue
            ytr, yte = torch.tensor(S[tr, bit:bit + 1]), torch.tensor(S[te, bit:bit + 1]); triv = max(S[te, bit].mean(), 1 - S[te, bit].mean())
            torch.manual_seed(seed * 100 + bit); rng = np.random.default_rng(seed * 100 + bit)
            L, d = [], 2 * n
            for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
            net = nn.Sequential(*L, nn.Linear(d, 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            first = -1; prev_ok = False; prev_t = 0
            for step in range(1, STEPS + 1):
                i = rng.integers(0, len(tr), BS)
                loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i]); opt.zero_grad(); loss.backward(); opt.step()
                if step % EV == 0:
                    with torch.no_grad(): acc = ((net(Xte) > 0).float() == yte).float().mean().item()
                    good = acc >= 0.99
                    if first < 0 and prev_ok and good: first = prev_t; break
                    prev_ok = good; prev_t = step
            w.writerow([q, bit, seed, first, acc, triv]); fh.flush(); print(q, bit, seed, first, round(acc, 4), round(float(triv), 4), flush=True)
