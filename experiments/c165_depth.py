# C-165: C-116 at other depths (env DEPTH hidden layers; arm "conf3" here means: the other outputs reach only the LAST hidden layer and the heads). DEPTH=3 reproduces C-116. Env B2 sets Adam's beta2 (C-166; default 0.999 = PyTorch default). OPT=rmsprop|adagrad (C-167, PyTorch defaults otherwise). LN=1 (or NORM=ln) puts a LayerNorm before each ReLU (C-169); NORM=rms (divide by the root mean square, no centring, no gain) or NORM=center (subtract the mean only) for C-170. FIXNORM=1: hidden-layer weight rows keep their initial norms (C-171). Extra column: correlation of the per-example norm of the second-to-last hidden layer with S at step 250.
# Original C-116 header: does confining the other outputs to the last hidden layer remove the width and frozen-layer dependence of the joint delay? Plain Adam (shared moments).
# arms: joint (all outputs train everything), conf3 (other outputs' gradient reaches only layer 3 and the heads), alone (other outputs' gradient reaches only the heads).
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; EV = 50; BS = 256; CAP = int(os.environ.get("CAP", "10000")); DEPTH = int(os.environ.get("DEPTH", "3")); B2 = float(os.environ.get("B2", "0.999")); NORM = os.environ.get("NORM", "ln" if os.environ.get("LN", "0") == "1" else "none")
W = int(os.environ.get("W", "128")); CS = float(os.environ.get("CS", "1")); LR = float(os.environ.get("LR", "1e-3")); OPT = os.environ.get("OPT", "adam"); FW = int(os.environ.get("FW", "0")); SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]; ARMS = os.environ.get("ARMS", "alone conf3 joint").split()
class RMS(nn.Module):
    def forward(self, x): return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-5)
class Center(nn.Module):
    def forward(self, x): return x - x.mean(-1, keepdim=True)
class FixedNormLinear(nn.Module):
    """Linear layer whose weight rows keep their initial norms: W = c * v / ||v|| (row-wise), v trained, c fixed (C-171)."""
    def __init__(self, lin):
        super().__init__(); self.v = nn.Parameter(lin.weight.detach().clone()); self.bias = nn.Parameter(lin.bias.detach().clone())
        self.register_buffer("c", lin.weight.detach().norm(dim=1, keepdim=True).clone())
    def forward(self, x): return nn.functional.linear(x, self.c * self.v / self.v.norm(dim=1, keepdim=True), self.bias)
FIXNORM = os.environ.get("FIXNORM", "0") == "1"
def lin(i, o): l = nn.Linear(i, o); return FixedNormLinear(l) if FIXNORM else l
NORM_LAYERS = [int(x) for x in os.environ.get("NORM_LAYERS", "").split(",") if x]  # 1-based hidden layers to normalise (empty = all), C-176
def normlayer(W): return [nn.LayerNorm(W)] if NORM == "ln" else [RMS()] if NORM == "rms" else [Center()] if NORM == "center" else []
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1); SUMV = (A + B).astype(np.float64)
def r2_on(h, s):
    hc = h - h.mean(0); sc = (s - s.mean()) / s.std(); beta = (hc * sc[:, None]).mean(0); return float((beta ** 2).sum() / hc.var(0).sum())
m_oth = torch.ones(n + 1, dtype=torch.bool); m_oth[4] = False
POW = torch.tensor([2.0 ** i for i in range(n)]); PROF = torch.tensor([float(v) for v in open('c187_profile.txt').read().split()]) if os.path.exists('c187_profile.txt') else None
fp = f"{OUT}/confine.csv"; new = not os.path.exists(fp); fh = open(fp, "a", newline=""); w = csv.writer(fh)
if new: w.writerow(["W", "FW", "CS", "seed", "arm", "t4", "L2_R2S250", "L3_R2S250", "L2norm_corrS250", "L3norm_corrS250"])
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    Xtr, Xte, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]), torch.tensor(Y[te])
    for arm in ARMS:
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4)
        if FW:
            frozen = nn.Sequential(nn.Linear(16, FW), nn.ReLU())
            for p in frozen.parameters(): p.requires_grad_(False)
            d = FW
        else: frozen = nn.Identity(); d = 16
        Ls = [nn.Sequential(lin(d if k == 0 else W, W), *(normlayer(W) if (not NORM_LAYERS or k + 1 in NORM_LAYERS) else []), nn.ReLU()) for k in range(DEPTH)]; head = nn.Linear(W, n + 1)
        low = nn.Sequential(*Ls[:-1]) if DEPTH > 1 else nn.Identity(); top = Ls[-1]
        if CS != 1.0:
            with torch.no_grad():
                for p in [p for m in (*Ls, head) for p in m.parameters()]: p.mul_(CS)
        C4 = float(arm[len("joint_4lr"):]) if arm.startswith("joint_4lr") else 1.0  # C-192: bit 4's read-out reparametrised as z4 = C4 * (w4 . h3 + b4), same initial logits; under Adam this scales its effective step size by C4
        w4 = nn.Parameter(head.weight.detach()[4].clone() / C4); b4 = nn.Parameter(head.bias.detach()[4:5].clone() / C4)
        params = [p for m in (*Ls, head) for p in m.parameters()] + ([w4, b4] if arm.startswith("joint_4lr") else []); opt = torch.optim.SGD(params, lr=LR, momentum=0.9) if OPT == "sgd" else torch.optim.RMSprop(params, lr=LR) if OPT == "rmsprop" else torch.optim.Adagrad(params, lr=LR) if OPT == "adagrad" else torch.optim.Adam(params, lr=LR, betas=(0.9, B2))
        def fwd(x):
            hl = low(frozen(x)); h3 = top(hl); z = head(h3)
            if arm == "joint": return z
            if arm in ("joint_orms", "joint_4rms"):  # C-177: RMS-normalised last hidden layer for the other outputs' heads only, or for bit 4's head only
                zr = head(h3 / torch.sqrt((h3 ** 2).mean(-1, keepdim=True) + 1e-5))
                return torch.where(m_oth, zr, z) if arm == "joint_orms" else torch.where(m_oth, z, zr)
            if arm in ("joint_4mag", "alone_4mag"):  # C-187: bit 4's head reads the last layer multiplied by a fixed U-shaped function of S (the C-180 profile, mean 1)
                Sx = (x[:, :n] * POW).sum(1) + (x[:, n:2 * n] * POW).sum(1); zm = head(h3 * PROF[torch.clamp((Sx / 32).long(), max=15)][:, None])
                if arm == "joint_4mag": return torch.where(m_oth, z, zm)
                return torch.where(m_oth, head(h3.detach()), zm)
            if arm.startswith("joint_4lr"):
                z4 = C4 * (h3 @ w4 + b4); return torch.where(m_oth, z, z4[:, None].expand_as(z))
            if arm == "joint_4glob":  # C-190: bit 4's head reads h3 divided by ONE number per batch (the batch root mean square, as a constant in the backward pass)
                zg = head(h3 / torch.sqrt((h3 ** 2).mean() + 1e-5).detach()); return torch.where(m_oth, z, zg)
            if arm == "joint_4divS":  # C-189: bit 4's head reads h3 divided by the fixed S-profile of the norm (C-180), not by the actual norm
                Sx = (x[:, :n] * POW).sum(1) + (x[:, n:2 * n] * POW).sum(1); zd = head(h3 / PROF[torch.clamp((Sx / 32).long(), max=15)][:, None]); return torch.where(m_oth, z, zd)
            if arm == "joint_4rms_sg":  # C-188: forward like joint_4rms, but the rms is a constant in the backward pass
                zs = head(h3 / torch.sqrt((h3 ** 2).mean(-1, keepdim=True) + 1e-5).detach()); return torch.where(m_oth, z, zs)
            if arm == "joint_4center":  # C-178: bit 4's head reads the mean-centred (not rescaled) last hidden layer
                zc = head(h3 - h3.mean(-1, keepdim=True)); return torch.where(m_oth, z, zc)
            zo = head(top(hl.detach())) if arm == "conf3" else head(h3.detach())
            return torch.where(m_oth, zo, z)
        first = None; prev_ok = False; prev_t = 0; r2 = [float("nan")] * 4
        for step in range(1, CAP + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step == 250:
                with torch.no_grad(): h2 = low(frozen(Xtr[:8192])); h3 = top(h2)
                r2 = [r2_on(h2.numpy().astype(np.float64), SUMV[tr][:8192]), r2_on(h3.numpy().astype(np.float64), SUMV[tr][:8192]), float(np.corrcoef(np.linalg.norm(h2.numpy().astype(np.float64), axis=1), SUMV[tr][:8192])[0, 1]), float(np.corrcoef(np.linalg.norm(h3.numpy().astype(np.float64), axis=1), SUMV[tr][:8192])[0, 1])]
            if step % EV == 0:
                with torch.no_grad(): acc = ((fwd(Xte)[:, 4] > 0).float() == Yte[:, 4]).float().mean().item()
                good = acc >= 0.99
                if prev_ok and good and first is None: first = prev_t
                prev_ok = good; prev_t = step
                if first is not None and step >= 250: break
        w.writerow([W, FW, CS, seed, arm, first if first is not None else -1, round(r2[0], 4), round(r2[1], 4), round(r2[2], 4), round(r2[3], 4)]); fh.flush(); print(W, FW, seed, arm, first, [round(x, 3) for x in r2], flush=True)
