# C-108: which layer's stored second moment slows the recovery, and how does the delay depend on its size? Custom Adam with per-parameter second-moment counters; first moment always fresh.
import sys, os, copy, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 3500; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False
class MyAdam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = list(params); self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [torch.zeros_like(x) for x in self.p]; self.v = [torch.zeros_like(x) for x in self.p]; self.tm = 0; self.tv = [0] * len(self.p); self.vscale = None
    def zero_grad(self):
        for x in self.p: x.grad = None
    @torch.no_grad()
    def step(self):
        self.tm += 1
        for k, (x, m, v) in enumerate(zip(self.p, self.m, self.v)):
            self.tv[k] += 1
            g = x.grad; m.mul_(self.b1).add_(g, alpha=1 - self.b1); v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            x.addcdiv_(m / (1 - self.b1 ** self.tm), (v / (1 - self.b2 ** self.tv[k])).sqrt() + self.eps, value=-self.lr)
def make():
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L), nn.Linear(d, n + 1)
def fwd(body, head, x, block):
    h = body(x); o_full = head(h); o_oth = head(h.detach()) if block else o_full
    return torch.where(m_oth, o_oth, o_full)
fp = f"{OUT}/adamparts.csv"; fh = open(fp, "a", newline=""); w = csv.writer(fh)
if os.path.getsize(fp) == 0: w.writerow(["seed", "branch", "extra_steps_to_0.99_bit4"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    body, head = make(); ps = list(body.parameters()) + list(head.parameters()); opt = MyAdam(ps)
    for step in range(1, 501):
        i = rng.integers(0, len(tr), BS)
        loss = nn.functional.binary_cross_entropy_with_logits(head(body(Xtr[i])), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    st = dict(body=copy.deepcopy(body.state_dict()), head=copy.deepcopy(head.state_dict()), m=[x.clone() for x in opt.m], v=[x.clone() for x in opt.v], t=opt.tm, rng=copy.deepcopy(rng.bit_generator.state))
    # layers: parameter index groups of (body L1 = 0,1; L2 = 2,3; L3 = 4,5; head = 6,7)
    groups = {"L1": [0, 1], "L2": [2, 3], "L3": [4, 5], "head": [6, 7]}
    arms = [("all_v", groups["L1"] + groups["L2"] + groups["L3"] + groups["head"], 1.0), ("none", [], 1.0), ("L1_v", groups["L1"], 1.0), ("L2_v", groups["L2"], 1.0), ("L3_v", groups["L3"], 1.0), ("head_v", groups["head"], 1.0),
            ("all_v_x0.1", groups["L1"] + groups["L2"] + groups["L3"] + groups["head"], 0.1), ("all_v_x0.3", groups["L1"] + groups["L2"] + groups["L3"] + groups["head"], 0.3), ("all_v_x3", groups["L1"] + groups["L2"] + groups["L3"] + groups["head"], 3.0)]
    for tag, keep_idx, vsc in arms:
        b2_, h2_ = make(); b2_.load_state_dict(copy.deepcopy(st["body"])); h2_.load_state_dict(copy.deepcopy(st["head"]))
        o = MyAdam(list(b2_.parameters()) + list(h2_.parameters()))
        for k in keep_idx:
            o.v[k] = st["v"][k].clone() * vsc; o.tv[k] = st["t"]
        r2 = np.random.default_rng(); r2.bit_generator.state = copy.deepcopy(st["rng"]); first = None; prev_ok = False; prev_t = 0
        for step in range(1, CAP + 1):
            i = r2.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(b2_, h2_, Xtr[i], True), Ytr[i], reduction="none").sum(1).mean(); o.zero_grad(); loss.backward(); o.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(b2_, h2_, Xte, True)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good: first = prev_t; break
                prev_ok = good; prev_t = step
        w.writerow([seed, tag, first if first is not None else -1]); fh.flush(); print(seed, tag, first, flush=True)
