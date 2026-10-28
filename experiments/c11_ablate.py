"""C-11: causal ablation of probe directions. Usage: python3 c11_ablate.py OUTDIR [--smoke]; env SEEDS="20 21 22". Resumable per (op, seed) via saved models and results file."""
import sys, os, csv, time
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv; os.makedirs(OUT, exist_ok=True)
n = 4 if SMOKE else 8; STEPS = 500 if SMOKE else 5000; WIDTH, DEPTH, LR, BS = 128, 3, 1e-3, 256; torch.set_num_threads(2)
SEEDS = [20] if SMOKE else [int(s) for s in os.environ.get("SEEDS", "20 21 22").split()]
a, b = np.meshgrid(np.arange(2 ** n), np.arange(2 ** n), indexing="ij"); a, b = a.ravel(), b.ravel()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
X = np.concatenate([bits(a, n), bits(b, n)], 1); Xt = torch.tensor(X)
c = np.zeros((len(a), n + 1), dtype=np.int64)
for i in range(n):
    ai, bi = (a >> i) & 1, (b >> i) & 1; c[:, i + 1] = (ai & bi) | (c[:, i] & (ai ^ bi))
rf = f"{OUT}/ablation.csv"; done = set()
if os.path.exists(rf): done = {(r["op"], r["seed"]) for r in csv.DictReader(open(rf))}
new = not os.path.exists(rf); fh = open(rf, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["op", "seed", "layer", "i", "control", "out_bit", "acc_base", "acc_abl", "acc_base_c1", "acc_abl_c1", "acc_base_c0", "acc_abl_c0"])
def layers(m):
    mods = list(m[0]); return [nn.Sequential(*mods[2 * k: 2 * k + 2]) for k in range(DEPTH)], m[1]
for seed in SEEDS:
    rng = np.random.default_rng(seed); perm = rng.permutation(len(a)); tr, te = perm[: len(a) // 2], perm[len(a) // 2:]
    pr = rng.permutation(len(a))[:10000]; ptr = pr[:5000]
    for op in ("add", "xor"):
        if (op, str(seed)) in done: continue
        Y = torch.tensor(bits(a + b, n + 1) if op == "add" else bits(a ^ b, n)); torch.manual_seed(seed)
        L, d = [], 2 * n
        for _ in range(DEPTH): L += [nn.Linear(d, WIDTH), nn.ReLU()]; d = WIDTH
        body = nn.Sequential(*L); head = nn.Linear(d, Y.shape[1]); model = nn.Sequential(body, head)
        mp = f"{OUT}/model_{op}_{seed}.pt"
        if os.path.exists(mp): model.load_state_dict(torch.load(mp))
        else:
            opt = torch.optim.Adam(model.parameters(), lr=LR); g = torch.Generator().manual_seed(1000 + seed); tri = torch.tensor(tr)
            for step in range(STEPS):
                ix = tri[torch.randint(0, len(tri), (BS,), generator=g)]
                loss = nn.functional.binary_cross_entropy_with_logits(model(Xt[ix]), Y[ix], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            torch.save(model.state_dict(), mp)
        blocks = [nn.Sequential(body[2 * k], body[2 * k + 1]) for k in range(DEPTH)]
        te_t = torch.tensor(te); Yte = Y[te_t]
        def run_from(h, k):
            for kk in range(k, DEPTH): h = blocks[kk](h)
            return head(h)
        with torch.no_grad():
            hs = [Xt]
            for blk in blocks: hs.append(blk(hs[-1]))
        def evaluate(k, wvec, i):
            """ablate direction wvec at hidden layer k (1-based) and return per-bit accuracies overall, c_i=1, c_i=0"""
            hh = hs[k][te_t]; u = torch.tensor(wvec / (np.linalg.norm(wvec) + 1e-12), dtype=torch.float32)
            with torch.no_grad():
                out = run_from(hh - (hh @ u)[:, None] * u[None, :], k)
            ok = ((out > 0).float() == Yte).float().numpy(); cc = c[te, i]
            return ok.mean(0), ok[cc == 1].mean(0), ok[cc == 0].mean(0)
        with torch.no_grad():
            base_out = model(Xt[te_t]); okb = ((base_out > 0).float() == Yte).float().numpy()
        for k in range(1, DEPTH + 1):
            feats = hs[k].numpy()
            for i in range(2, n + 1):
                cc = c[te, i]; b_all, b1, b0 = okb.mean(0), okb[cc == 1].mean(0), okb[cc == 0].mean(0)
                dirs = {}
                for name, tgt in (("carry", c[:, i]), ("inputbit", (a >> (i - 1)) & 1)):
                    clf = LogisticRegression(max_iter=200, C=10.0).fit(feats[ptr], tgt[ptr]); dirs[name] = clf.coef_[0]
                rr = np.random.default_rng(seed * 100 + k * 10 + i)
                for name in ("carry", "inputbit"):
                    r_all, r1, r0 = evaluate(k, dirs[name], i)
                    for j in range(Y.shape[1]): w.writerow([op, seed, k, i, name, j, f"{b_all[j]:.5f}", f"{r_all[j]:.5f}", f"{b1[j]:.5f}", f"{r1[j]:.5f}", f"{b0[j]:.5f}", f"{r0[j]:.5f}"])
                rs = [evaluate(k, rr.normal(size=WIDTH), i) for _ in range(20)]
                m_all, m1, m0 = (np.mean([r[q] for r in rs], 0) for q in range(3))
                for j in range(Y.shape[1]): w.writerow([op, seed, k, i, "random", j, f"{b_all[j]:.5f}", f"{m_all[j]:.5f}", f"{b1[j]:.5f}", f"{m1[j]:.5f}", f"{b0[j]:.5f}", f"{m0[j]:.5f}"])
        fh.flush(); print(f"{op} seed={seed} done", flush=True)
print("done")
