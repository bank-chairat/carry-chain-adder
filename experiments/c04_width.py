"""C-04: smallest width for exact fit of addition vs XOR. Usage: python3 c04_width.py OUTDIR [--smoke]. Resumable, one row per run."""
import sys, os, csv, itertools, time
import numpy as np, torch, torch.nn as nn
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv; os.makedirs(OUT, exist_ok=True); torch.set_num_threads(1)
NS = (3,) if SMOKE else tuple(int(x) for x in os.environ.get("NS", "3 4 5 6").split());
WIDTHS = (4, 8) if SMOKE else (2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128)
SEEDS = (40,) if SMOKE else range(40, 45); STEPS = 500 if SMOKE else 20000
f = f"{OUT}/runs.csv"; done = set()
if os.path.exists(f): done = {tuple(r[k] for k in ("op", "n", "depth", "width", "seed")) for r in csv.DictReader(open(f))}
new = not os.path.exists(f); fh = open(f, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["op", "n", "depth", "width", "seed", "success", "final_wrong_pairs", "steps_to_exact"])
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
for n in NS:
    a, b = np.meshgrid(np.arange(2 ** n), np.arange(2 ** n), indexing="ij"); a, b = a.ravel(), b.ravel()
    X = torch.tensor(np.concatenate([bits(a, n), bits(b, n)], 1))
    for op in ("add", "xor"):
        Y = torch.tensor(bits(a + b, n + 1) if op == "add" else bits(a ^ b, n))
        for depth in (1, 2):
            for width in WIDTHS:
                for seed in SEEDS:
                    key = (op, str(n), str(depth), str(width), str(seed))
                    if key in done: continue
                    torch.manual_seed(seed); L, d = [], 2 * n
                    for _ in range(depth): L += [nn.Linear(d, width), nn.ReLU()]; d = width
                    m = nn.Sequential(*L, nn.Linear(d, Y.shape[1])); opt = torch.optim.Adam(m.parameters(), lr=1e-2); ex = -1
                    for s in range(1, STEPS + 1):
                        out = m(X); loss = nn.functional.binary_cross_entropy_with_logits(out, Y); opt.zero_grad(); loss.backward(); opt.step()
                        if s % 100 == 0:
                            wrong = int((((out > 0).float() != Y).any(1)).sum())
                            if wrong == 0: ex = s; break
                    with torch.no_grad(): wrong = int((((m(X) > 0).float() != Y).any(1)).sum())
                    w.writerow([op, n, depth, width, seed, int(wrong == 0), wrong, ex]); fh.flush()
        print(f"n={n} {op} done", flush=True)
print("done")
