# C-153: reproduce the returned note's forcing branches in my own code. After 500 joint steps (plain Adam), the body is trained on bit 4's gradient plus one of:
# joint: the others' current gradient h(theta,B); block: nothing; frozen_other: the others' gradient at the frozen step-500 parameters on the same minibatch h_ref(B);
# centered_noise: h_ref(B) - h_bar (h_bar = others' mean gradient at step 500 over all training pairs); frozen_mean: h_bar. Heads always get the ordinary joint gradient; Adam state inherited.
import sys, os, copy, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 3500; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
BR = os.environ.get("BRANCHES", "block centered_noise frozen_mean frozen_other joint").split()
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); oth = [k for k in range(n + 1) if k != 4]
def make():
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L), nn.Linear(d, n + 1)
fp = f"{OUT}/forcing.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["seed", "branch", "extra_steps_to_0.99_bit4"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4); body, head = make()
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
    for step in range(1, 501):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(head(body(Xtr[i])), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    ref_body, ref_head = copy.deepcopy(body), copy.deepcopy(head); rb = list(ref_body.parameters())
    hbar = [torch.zeros_like(p) for p in rb]
    for c0 in range(0, len(tr), 1024):
        per = nn.functional.binary_cross_entropy_with_logits(ref_head(ref_body(Xtr[c0:c0 + 1024])), Ytr[c0:c0 + 1024], reduction="none")[:, oth].sum()
        for a, g in zip(hbar, torch.autograd.grad(per, rb)): a.add_(g / len(tr))
    st = dict(body=copy.deepcopy(body.state_dict()), head=copy.deepcopy(head.state_dict()), opt=copy.deepcopy(opt.state_dict()), rng=copy.deepcopy(rng.bit_generator.state))
    for br in BR:
        b2, h2 = make(); b2.load_state_dict(copy.deepcopy(st["body"])); h2.load_state_dict(copy.deepcopy(st["head"])); bp, hp = list(b2.parameters()), list(h2.parameters())
        o2 = torch.optim.Adam(bp + hp, lr=1e-3); o2.load_state_dict(copy.deepcopy(st["opt"]))
        r2 = np.random.default_rng(); r2.bit_generator.state = copy.deepcopy(st["rng"]); first = None; prev_ok = False; prev_t = 0
        for step in range(1, CAP + 1):
            i = r2.integers(0, len(tr), BS); per = nn.functional.binary_cross_entropy_with_logits(h2(b2(Xtr[i])), Ytr[i], reduction="none").mean(0)
            g4 = torch.autograd.grad(per[4], bp, retain_graph=True); ghead = torch.autograd.grad(per.sum(), hp, retain_graph=True)
            if br == "joint": extra = torch.autograd.grad(per[oth].sum(), bp)
            elif br == "block": extra = [torch.zeros_like(p) for p in bp]
            else:
                if br != "frozen_mean":
                    pr = nn.functional.binary_cross_entropy_with_logits(ref_head(ref_body(Xtr[i])), Ytr[i], reduction="none").mean(0)[oth].sum(); href = torch.autograd.grad(pr, rb)
                extra = {"frozen_other": lambda: href, "centered_noise": lambda: [a - b for a, b in zip(href, hbar)], "frozen_mean": lambda: hbar}[br]()
            for p, a, b in zip(bp, g4, extra): p.grad = a + b
            for p, g in zip(hp, ghead): p.grad = g
            o2.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((h2(b2(Xte))[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([seed, br, first if first is not None else -1]); fh.flush(); print(seed, br, first, flush=True)
