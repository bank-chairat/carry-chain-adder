import sys, os, csv, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore"); torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; BS = 256; STEPS = int(os.environ.get('STEPS', '4000'))
NETS = [('single', [4]), ('lower', [0, 1, 2, 3, 4]), ('upper', [4, 5, 6, 7, 8]), ('joint', list(range(9)))]
SEEDS = [int(x) for x in os.environ.get('SEEDS', '0 1 2 3 4').split()]
CKPTS = set(list(range(0, 2001, 50)) + list(range(2100, 4001, 100))) if STEPS == 4000 else set(range(0, STEPS + 1, 10))
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
T = {"p4": ((A ^ B) >> 4) & 1, "c4": (((A & 15) + (B & 15)) >> 4) & 1, "p3": ((A ^ B) >> 3) & 1}
fp = f"{OUT}/probe_time.csv"; done = set()
if os.path.exists(fp): done = {(r["net"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["net", "seed", "step", "bit4_acc", "target", "layer", "probe_acc"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    pr = np.random.default_rng(700 + seed); ptr = pr.choice(tr, 5000, replace=False); pte = pr.choice(te, 5000, replace=False)
    Xtr, Xte = torch.tensor(X[tr]), torch.tensor(X[te])
    for kind, outs in NETS:
        if (kind, str(seed)) in done: continue
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        Ytr = torch.tensor(S[tr][:, outs]); out = len(outs); col = outs.index(4)
        layers = []; d = 2 * n
        for _ in range(3): layers.append(nn.Sequential(nn.Linear(d, 128), nn.ReLU())); d = 128
        head = nn.Linear(d, out); params = [p for l in layers for p in l.parameters()] + list(head.parameters()); opt = torch.optim.Adam(params, lr=1e-3)
        def fwd(x, acts=False):
            hs = []
            for l in layers: x = l(x); hs.append(x)
            return (head(x), hs) if acts else head(x)
        def probe(step):
            with torch.no_grad():
                _, hs_tr = fwd(torch.tensor(X[ptr]), True); _, hs_te = fwd(torch.tensor(X[pte]), True)
                lo = fwd(Xte)[:, col]; acc4 = float(((lo > 0).float() == torch.tensor(S[te][:, 4])).float().mean())
            for L in range(3):
                m_, s_ = hs_tr[L].numpy().mean(0), hs_tr[L].numpy().std(0) + 1e-6
                a_ = np.c_[(hs_tr[L].numpy() - m_) / s_, np.ones(len(ptr))]; b_ = np.c_[(hs_te[L].numpy() - m_) / s_, np.ones(len(pte))]
                G = np.linalg.inv(a_.T @ a_ + 1.0 * np.eye(a_.shape[1]))              # closed-form ridge probe (lambda = 1)
                for name, y in T.items():
                    wv = G @ (a_.T @ (2.0 * y[ptr] - 1.0)); pred = (b_ @ wv > 0).astype(int)
                    w.writerow([kind, seed, step, round(acc4, 4), name, L + 1, round(float((pred == y[pte]).mean()), 4)])
            fh.flush()
        for step in range(0, STEPS + 1):
            if step in CKPTS: probe(step)
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        print(kind, seed, "done", flush=True)
