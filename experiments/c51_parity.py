import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); EV = 50; BS = 256; STEPS = 20000; RS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
def sample(rng, m, r):
    x = (rng.random((m, 16)) < 0.5).astype(np.float32); x[:, 1] = (rng.random(m) < r).astype(np.float32)
    y = (x[:, 0].astype(int) ^ x[:, 1].astype(int)).astype(np.float32)[:, None]; return torch.tensor(x), torch.tensor(y)
fp = f"{OUT}/parity.csv"; done = set()
if os.path.exists(fp): done = {(r["r"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["r", "seed", "time_to_099", "final_acc"])
for seed in range(5):
    for r in RS:
        if (str(r), str(seed)) in done: continue
        torch.manual_seed(seed * 100); rng = np.random.default_rng(seed * 100 + int(r * 1000)); Xte, Yte = sample(np.random.default_rng(777 + seed), 20000, r)
        L, d = [], 16
        for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
        net = nn.Sequential(*L, nn.Linear(d, 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        first = -1; prev_ok = False; prev_t = 0
        for step in range(1, STEPS + 1):
            xb, yb = sample(rng, BS, r)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte) > 0).float() == Yte).float().mean().item()
                good = acc >= 0.99
                if first < 0 and prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([r, seed, first, acc]); fh.flush(); print(r, seed, first, round(acc, 4), flush=True)
