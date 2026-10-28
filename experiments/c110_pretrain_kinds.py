# C-110: what kind of early joint training makes the body a worse start for bit 4? Pretrain K=250 steps with different label sets (Adam 1e-3, batch 256), then reset the head to its initial values, fresh Adam, train the body on bit 4 only (all heads train) until bit-4 accuracy >= 0.99.
import sys, os, copy, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 3500; K = int(os.environ.get("K", "250")); SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False
def make():
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L), nn.Linear(d, n + 1)
def fwd(body, head, x, block):
    h = body(x); o_full = head(h); o_oth = head(h.detach()) if block else o_full
    return torch.where(m_oth, o_oth, o_full)
fp = f"{OUT}/kinds.csv"; fh = open(fp, "a", newline=""); w = csv.writer(fh)
if os.path.getsize(fp) == 0: w.writerow(["seed", "pretrain", "extra_steps_to_0.99_bit4", "acc0"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    sub = np.random.default_rng(900 + seed); masks = np.zeros((9, 2 * n), np.float32)
    for j in range(9): masks[j, sub.choice(2 * n, 4, replace=False)] = 1
    Yrand = torch.tensor((X @ masks.T) % 2)[tr]
    for kind in ("none", "all9", "others8", "bit4only", "random9"):
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        body, head = make(); init_head = copy.deepcopy(head.state_dict())
        if kind != "none":
            opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
            for step in range(1, K + 1):
                i = rng.integers(0, len(tr), BS); z = head(body(Xtr[i]))
                tgt = Yrand[i] if kind == "random9" else Ytr[i]
                msk = {"all9": torch.ones(n + 1), "random9": torch.ones(n + 1), "others8": m_oth.float(), "bit4only": (~m_oth).float()}[kind]
                loss = (nn.functional.binary_cross_entropy_with_logits(z, tgt, reduction="none") * msk).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        else:
            for step in range(1, K + 1): rng.integers(0, len(tr), BS)    # keep the minibatch stream aligned
        h2 = nn.Linear(128, n + 1); h2.load_state_dict(init_head)
        o2 = torch.optim.Adam(list(body.parameters()) + list(h2.parameters()), lr=1e-3)
        with torch.no_grad(): acc0 = ((fwd(body, h2, Xte, True)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
        first = None; prev_ok = False; prev_t = 0
        for step in range(1, CAP + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(body, h2, Xtr[i], True), Ytr[i], reduction="none").sum(1).mean(); o2.zero_grad(); loss.backward(); o2.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(body, h2, Xte, True)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([seed, kind, first if first is not None else -1, round(acc0, 3)]); fh.flush(); print(seed, kind, first, round(acc0, 3), flush=True)
