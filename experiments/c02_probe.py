"""C-02: linear probes for the carry in hidden layers of MLPs trained on addition and XOR (n = 8). Design in PROGRESS.md (C-02).
Usage: python3 c02_probe.py OUTDIR [--smoke]    (resumable per (op, seed); writes OUTDIR/probes.csv and OUTDIR/outbits.csv)"""
import sys, os, csv, time
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv; os.makedirs(OUT, exist_ok=True)
n = 4 if SMOKE else 8; SEEDS = (20,) if SMOKE else range(20, 23); STEPS = 500 if SMOKE else 5000; EVERY = 250 if SMOKE else 500
WIDTH, DEPTH, LR, BS = 128, 3, 1e-3, 256; torch.set_num_threads(4)
a, b = np.meshgrid(np.arange(2 ** n), np.arange(2 ** n), indexing="ij"); a, b = a.ravel(), b.ravel()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
X = np.concatenate([bits(a, n), bits(b, n)], 1)
c = np.zeros((len(a), n + 1), dtype=np.int64)
for i in range(n):
    ai, bi = (a >> i) & 1, (b >> i) & 1; c[:, i + 1] = (ai & bi) | (c[:, i] & (ai ^ bi))
def build():
    L, d = [], 2 * n
    for _ in range(DEPTH): L += [nn.Linear(d, WIDTH), nn.ReLU()]; d = WIDTH
    return nn.Sequential(*L), d
pp, op_ = f"{OUT}/probes.csv", f"{OUT}/outbits.csv"; done = set()
if os.path.exists(pp): done = {(r["op"], r["seed"]) for r in csv.DictReader(open(pp))}
newp, newo = not os.path.exists(pp), not os.path.exists(op_)
fp, fo = open(pp, "a", newline=""), open(op_, "a", newline=""); wp, wo = csv.writer(fp), csv.writer(fo)
if newp: wp.writerow(["op", "seed", "step", "layer", "carry_bit", "probe_acc", "majority_rate"])
if newo: wo.writerow(["op", "seed", "step", "bit", "test_acc"])
def probe_all(feats, target, step, op, seed, layer):
    ntr = len(target) // 2
    for i in range(1, n + 1):
        y = target[:, i]; maj = max(y.mean(), 1 - y.mean())
        if y[:ntr].min() == y[:ntr].max(): acc = float(y[ntr:].mean() == y[0]) if False else maj
        else:
            clf = LogisticRegression(max_iter=200, C=10.0).fit(feats[:ntr], y[:ntr]); acc = clf.score(feats[ntr:], y[ntr:])
        wp.writerow([op, seed, step, layer, i, f"{acc:.4f}", f"{maj:.4f}"])
for seed in SEEDS:
    rng = np.random.default_rng(seed); perm = rng.permutation(len(a)); tr, te = perm[: len(a) // 2], perm[len(a) // 2:]
    pr = rng.permutation(len(a))[:10000]                                              # pairs used for probing
    for op in ("add", "xor"):
        if (op, str(seed)) in done: continue
        t0 = time.time(); Y = bits(a + b, n + 1) if op == "add" else bits(a ^ b, n)
        Xt, Yt = torch.tensor(X), torch.tensor(Y); torch.manual_seed(seed)
        body, d = build(); head = nn.Linear(d, Y.shape[1]); model = nn.Sequential(body, head)
        opt = torch.optim.Adam(model.parameters(), lr=LR); g = torch.Generator().manual_seed(1000 + seed); tri = torch.tensor(tr); tei = torch.tensor(te)
        def checkpoint(step):
            with torch.no_grad():
                acc = ((model(Xt[tei]) > 0).float() == Yt[tei]).float().mean(0).numpy()
                for k in range(Y.shape[1]): wo.writerow([op, seed, step, k, f"{acc[k]:.5f}"])
                h = Xt[pr]
                if step == 0: probe_all(X[pr], c[pr], step, op, seed, "input")
                for li in range(DEPTH):
                    h = body[2 * li + 1](body[2 * li](h)); probe_all(h.numpy(), c[pr], step, op, seed, f"h{li + 1}")
        checkpoint(0)
        for step in range(1, STEPS + 1):
            ix = tri[torch.randint(0, len(tri), (BS,), generator=g)]
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xt[ix]), Yt[ix], reduction="none").sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % EVERY == 0: checkpoint(step)
        fp.flush(); fo.flush(); print(f"{op} seed={seed} done ({time.time() - t0:.0f}s)", flush=True)
print("done")
