import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 25; BS = 256; STEPS = 3000
NS = int(os.environ.get("NS", "60")); START = int(os.environ.get("START", "0"))
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y4 = bits(A + B, n + 1)[:, 4:5]
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xall = torch.tensor(X); Xtr, Xte = Xall[tr], Xall[te]; ytr, yte = torch.tensor(Y4[tr]), torch.tensor(Y4[te])
tp4 = ((1 - 2 * ((A >> 4) & 1)) * (1 - 2 * ((B >> 4) & 1))).astype(np.float64); tctl = ((1 - 2 * ((A >> 2) & 1)) * (1 - 2 * ((B >> 6) & 1))).astype(np.float64)
pr = np.random.default_rng(700); ptr = pr.choice(tr, 20000, replace=False); pte = pr.choice(te, 20000, replace=False); yp = tp4 > 0
def corr_max(H, t):
    Hc = H - H.mean(0); tc = t - t.mean(); den = np.sqrt((Hc ** 2).sum(0) * (tc ** 2).sum()) + 1e-12
    return float(np.abs((Hc * tc[:, None]).sum(0) / den).max())
fp = f"{OUT}/init_align_n8.csv"; done = set()
if os.path.exists(fp): done = {r["s"] for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["s", "time", "A_p", "A_ctl", "probe_best", "probe_l1", "probe_l3"])
for s in range(START, START + NS):
    if str(s) in done: continue
    torch.manual_seed(10000 + s); rng = np.random.default_rng(777)
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, 1))
    with torch.no_grad():
        h1 = net[1](net[0](Xall)).numpy().astype(np.float64); Ap = corr_max(h1, tp4); Actl = corr_max(h1, tctl)
        accs = []
        for Lk in (1, 3, 5):
            def act(idx):
                x = Xall[idx]
                for q in range(Lk + 1): x = net[q](x)
                return x.numpy()
            a_, b_ = act(ptr), act(pte); m_, s_ = a_.mean(0), a_.std(0) + 1e-6
            a_ = np.c_[(a_ - m_) / s_, np.ones(len(a_))]; b_ = np.c_[(b_ - m_) / s_, np.ones(len(b_))]
            G = np.linalg.inv(a_.T @ a_ + np.eye(a_.shape[1])); wv = G @ (a_.T @ (2.0 * yp[ptr] - 1)); accs.append(float(((b_ @ wv > 0) == yp[pte]).mean()))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); first = -1; prev_ok = False; prev_t = 0
    for step in range(1, STEPS + 1):
        i = rng.integers(0, len(tr), BS)
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i]); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((net(Xte) > 0).float() == yte).float().mean().item()
            good = acc >= 0.99
            if first < 0 and prev_ok and good: first = prev_t
            prev_ok = good; prev_t = step
            if first >= 0: break
    w.writerow([s, first, round(Ap, 4), round(Actl, 4), round(max(accs), 4), round(accs[0], 4), round(accs[2], 4)]); fh.flush(); print(s, first, round(Ap, 3), round(Actl, 3), round(max(accs), 3), flush=True)
