# C-133b: iterative gradient ascent of J (bit-4 logit-target correlation after 125 joint steps) on the initial layer 1 of slow initialisations: STEPS steps of norm STEP, gradient recomputed each time; then normal training (C-83 recipe, float32).
import sys, os, csv, numpy as np, torch, torch.nn as nn
sys.argv = [sys.argv[0], "c133_scratch"]; os.environ.setdefault("INITS", "0")
exec(open("c133_attribution.py").read().split("for s in INITS:")[0])          # reuse definitions (init_params, f, J_of, data)
torch.set_num_threads(2)
NSTEP = int(os.environ.get("NSTEP", "4")); STEP = float(os.environ.get("STEP", "0.2")); OUT2 = "c133b"; os.makedirs(OUT2, exist_ok=True)
SL = [int(x) for x in os.environ["SLOW"].split()]
fp = f"{OUT2}/ascent.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["s", "J_path", "t4", "corr125_float32"])
for s in SL:
    P0 = init_params(s); W1, b1 = P0[0].clone(), P0[1].clone(); Js = []
    for k in range(NSTEP):
        Wv, bv = W1.clone().requires_grad_(True), b1.clone().requires_grad_(True); rest = [q.clone().requires_grad_(True) for q in P0[2:]]
        J = J_of([Wv, bv] + rest); gW, gb = torch.autograd.grad(J, [Wv, bv]); nrm = float(torch.sqrt((gW ** 2).sum() + (gb ** 2).sum()))
        Js.append(round(float(J.detach()), 4)); W1 = W1 + STEP * gW / nrm; b1 = b1 + STEP * gb / nrm
    Js.append(round(float(J_of([W1.clone().requires_grad_(True), b1.clone().requires_grad_(True)] + [q.clone().requires_grad_(True) for q in P0[2:]]).detach()), 4))
    # train in float32 from the moved initialisation
    torch.set_default_dtype(torch.float32); torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1))
    with torch.no_grad(): net[0].weight.copy_(W1.float()); net[0].bias.copy_(b1.float())
    Xtr32, Xte32, ytr32, yte32 = Xtr.float(), Xte.float(), ytr.float(), torch.tensor(Y[te]).float()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(777); first = None; prev_ok = False; prev_t = 0; c125 = float("nan")
    for step in range(1, 9001):
        i = rng.integers(0, len(tr), BS); loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr32[i]), ytr32[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step == 125:
            with torch.no_grad(): z = net(Xte32)[:, 4].numpy().astype(np.float64)
            c125 = float(np.corrcoef(z, t4.numpy())[0, 1])
        if step % 50 == 0:
            with torch.no_grad(): acc = ((net(Xte32)[:, 4] > 0).float() == yte32[:, 4]).float().mean().item()
            good = acc >= 0.99
            if prev_ok and good and first is None: first = prev_t
            prev_ok = good; prev_t = step
            if first is not None and step >= 125: break
    torch.set_default_dtype(torch.float64)
    w.writerow([s, " ".join(str(x) for x in Js), first if first is not None else -1, round(c125, 4)]); fh.flush(); print(s, Js, first, round(c125, 3), flush=True)
