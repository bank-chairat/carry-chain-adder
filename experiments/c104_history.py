# C-104: after 500 steps of joint training, train the shared body on bit 4 only (the other eight outputs' gradient into the body is blocked; all heads keep training) from five different starting states.
import sys, os, copy, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = 3500; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
others = torch.tensor([b for b in range(n + 1) if b != 4]); m_oth = torch.zeros(n + 1, dtype=torch.bool); m_oth[others] = True
def make():
    L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L), nn.Linear(d, n + 1)
def fwd(body, head, x, block):
    h = body(x); o_full = head(h); o_oth = head(h.detach()) if block else o_full
    return torch.where(m_oth, o_oth, o_full)
def train_branch(body, head, opt, rng, tr, Xtr, Ytr, Xte, Yte, block, steps, tag, w, seed):
    first = None; prev_ok = False; prev_t = 0
    for step in range(1, steps + 1):
        i = rng.integers(0, len(tr), BS)
        loss = nn.functional.binary_cross_entropy_with_logits(fwd(body, head, Xtr[i], block), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % EV == 0:
            with torch.no_grad(): acc = ((fwd(body, head, Xte, block)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
            good = acc >= 0.99
            if prev_ok and good and first is None: first = prev_t; break
            prev_ok = good; prev_t = step
    w.writerow([seed, tag, first if first is not None else -1]); print(seed, tag, first, flush=True)
fp = f"{OUT}/history.csv"; fh = open(fp, "a", newline=""); w = csv.writer(fh)
if os.path.getsize(fp) == 0: w.writerow(["seed", "branch", "extra_steps_to_0.99_bit4"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
    body, head = make(); init_body, init_head = copy.deepcopy(body.state_dict()), copy.deepcopy(head.state_dict())
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
    for step in range(1, 501):
        i = rng.integers(0, len(tr), BS)
        loss = nn.functional.binary_cross_entropy_with_logits(head(body(Xtr[i])), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    cur_body, cur_head, cur_opt, rng_state = copy.deepcopy(body.state_dict()), copy.deepcopy(head.state_dict()), copy.deepcopy(opt.state_dict()), copy.deepcopy(rng.bit_generator.state)
    for tag in ("current_inherited", "current_freshAdam", "initial_freshAdam", "curbody_inithead_freshAdam", "initbody_curhead_freshAdam"):
        bsd, hsd = {"current_inherited": (cur_body, cur_head), "current_freshAdam": (cur_body, cur_head), "initial_freshAdam": (init_body, init_head),
                    "curbody_inithead_freshAdam": (cur_body, init_head), "initbody_curhead_freshAdam": (init_body, cur_head)}[tag]
        b2, h2 = make(); b2.load_state_dict(copy.deepcopy(bsd)); h2.load_state_dict(copy.deepcopy(hsd))
        o2 = torch.optim.Adam(list(b2.parameters()) + list(h2.parameters()), lr=1e-3)
        if tag == "current_inherited": o2.load_state_dict(copy.deepcopy(cur_opt))
        r2 = np.random.default_rng(); r2.bit_generator.state = copy.deepcopy(rng_state)
        train_branch(b2, h2, o2, r2, tr, Xtr, Ytr, Xte, Yte, True, CAP, tag, w, seed); fh.flush()
