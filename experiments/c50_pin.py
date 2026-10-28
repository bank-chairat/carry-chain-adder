"""C-14: hump at n = 12, 16 with Adam / momentum SGD, strict held-out panel (pairs and swaps excluded from training), 20k steps.
Usage: N=12 OPT=adam TASK=add SEEDS="0 1" python3 c14_scale.py OUTDIR  (one thread; resumable per run)"""
import sys, os, csv, time
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
n = int(os.environ.get("N", "12")); OPT = os.environ.get("OPT", "adam"); TASK = os.environ.get("TASK", "add"); STEPS = int(os.environ.get("STEPS", "20000"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7 8 9").split()]; EV = 200; BS = 256
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
def target(a, b): return bits(a + b, n + 1) if TASK == "add" else bits(a ^ b, n + 1)   # XOR padded with a constant-zero overflow bit
erng = np.random.default_rng(70000 + n); ea = erng.integers(0, 1 << n, 8192); eb = erng.integers(0, 1 << n, 8192)
excl = np.unique(np.concatenate([(ea << n) | eb, (eb << n) | ea]))
PIN = os.environ.get('PIN') == '1'
def feat(a, b): return np.concatenate([bits(a, n), bits(b, n)] + ([bits(a ^ b, n)] if PIN else []), 1)
Xe = torch.tensor(feat(ea, eb)); Ye = torch.tensor(target(ea, eb))
fp = f"{OUT}/runs_{TASK}_{OPT}_n{n}_pin.csv"; done = set()
if os.path.exists(fp): done = {r["seed"] for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["task", "opt", "n", "seed", "bit_times_json", "final_acc_json"])
import json
for seed in SEEDS:
    if str(seed) in done: continue
    t0 = time.time(); rng = np.random.default_rng(1000 + seed); torch.manual_seed(seed)
    L, d = [], (3 if PIN else 2) * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3) if OPT == "adam" else torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    first = np.full(n + 1, -1); prev_ok = np.zeros(n + 1, bool); prev_t = 0
    for step in range(1, STEPS + 1):
        a = rng.integers(0, 1 << n, BS * 2); b = rng.integers(0, 1 << n, BS * 2)
        ok = ~np.isin((a << n) | b, excl); a, b = a[ok][:BS], b[ok][:BS]
        x = torch.tensor(feat(a, b)); y = torch.tensor(target(a, b))
        loss = nn.functional.binary_cross_entropy_with_logits(net(x), y, reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xe) > 0).float() == Ye).float().mean(0).numpy()
            good = acc >= 0.99
            newly = (first < 0) & prev_ok & good          # persisted: good at previous and this evaluation
            first[newly] = prev_t; prev_ok = good; prev_t = step
    w.writerow([TASK, OPT, n, seed, json.dumps(first.tolist()), json.dumps(acc.tolist())]); fh.flush()
    print(f"{TASK} {OPT} n={n} seed={seed} times={first.tolist()} ({time.time() - t0:.0f}s)", flush=True)
