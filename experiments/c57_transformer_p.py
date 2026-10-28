import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 100; BS = 256; STEPS = 20000; LRS = [1e-3]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n), bits(A ^ B, n)], 1); Y = bits(A + B, n + 1)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.val = nn.Embedding(2, 64); s.pos = nn.Parameter(torch.randn(3 * n, 64) * 0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True, norm_first=True), 2); s.out = nn.Linear(64, n + 1)
    def forward(s, x): return s.out(s.enc(s.val(x.long()) + s.pos).mean(1))
fp = f"{OUT}/transformer_p_n8.csv"; done = set()
if os.path.exists(fp): done = {(r["lr"], r["seed"]) for r in csv.DictReader(open(fp))}
new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["lr", "seed"] + [f"t{b}" for b in range(n + 1)] + ["final_acc_mean"])
for lr in LRS:
    for seed in range(5):
        if (str(lr), str(seed)) in done: continue
        perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
        Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4); net = Net(); opt = torch.optim.Adam(net.parameters(), lr=lr)
        first = np.full(n + 1, -1); prev_ok = np.zeros(n + 1, bool); prev_t = 0
        for step in range(1, STEPS + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xte) > 0).float() == Yte).float().mean(0).numpy()
                good = acc >= 0.99; newly = (first < 0) & prev_ok & good; first[newly] = prev_t; prev_ok = good; prev_t = step
                if (first >= 0).all(): break
        w.writerow([lr, seed] + first.tolist() + [float(acc.mean())]); fh.flush(); print(lr, seed, first.tolist(), round(float(acc.mean()), 4), flush=True)
