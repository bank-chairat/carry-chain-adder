import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = torch.tensor(np.concatenate([bits(A, n), bits(B, n)], 1)); S = (A + B).astype(np.float64); Y = bits(A + B, n + 1)
P = ((A ^ B)[:, None] >> np.arange(8)) & 1; C = np.stack([(((A & ((1 << i) - 1)) + (B & ((1 << i) - 1))) >> i) & 1 for i in range(8)], 1)
def r2(H, D):
    D = np.c_[D, np.ones(len(D))]; beta = np.linalg.lstsq(D, H, rcond=None)[0]; res = H - D @ beta; v = H.var(0); out = 1 - res.var(0) / np.where(v > 1e-9, v, 1.0); out[v <= 1e-9] = 0.0; return out   # dead units get 0 (bug fix: first run gave them 1)
def table(i):
    p, c = P[:, i], C[:, i]; return np.c_[p * (1 - c), (1 - p) * c, p * c].astype(np.float64)
rng = np.random.default_rng(3)
for s in range(10):
    net = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 9))
    net.load_state_dict(torch.load(f'c91_nets/net_s{s}.pt', weights_only=True))
    with torch.no_grad(): H2 = net[:4](X).numpy().astype(np.float64); H3 = net[:6](X).numpy().astype(np.float64); base = ((net(X)[:, 4] > 0).numpy() == (Y[:, 4] > 0.5)).mean()
    live2, live3 = H2.var(0) > 1e-9, H3.var(0) > 1e-9
    rS2 = r2(H2, np.c_[S, S ** 2])[live2]; rS3 = r2(H3, np.c_[S, S ** 2])[live3]
    f4_3 = r2(H3, table(4)); f4_2 = r2(H2, table(4)); out3 = r2(H3, Y)[live3]
    n_f4_3 = int((f4_3 >= 0.5).sum()); n_f4_2 = int((f4_2 >= 0.5).sum())
    other = [int((r2(H3, table(i)) >= 0.5).sum()) for i in range(8)]
    idx = np.where(f4_3 >= 0.5)[0]
    def ablate(units):
        with torch.no_grad():
            h = net[:6](X).clone(); mu = h.mean(0); h[:, units] = mu[units]; return ((net[6:](h)[:, 4] > 0).numpy() == (Y[:, 4] > 0.5)).mean()
    if len(idx):
        d_f = base - ablate(list(idx)); cand = np.where(live3 & (f4_3 < 0.5))[0]; d_r = np.mean([base - ablate(list(rng.choice(cand, len(idx), replace=False))) for _ in range(5)])
    else: d_f = d_r = float('nan')
    print(f'net {s}: live L2 {live2.sum()} L3 {live3.sum()} | median R2 on S: L2 {np.median(rS2):.2f} L3 {np.median(rS3):.2f} | bit-4 factor units L2 {n_f4_2} L3 {n_f4_3} | L3 factor units per bit 0-7 {other} | median R2 L3 on 9 sum bits {np.median(out3):.2f} | ablate bit-4 factor units: drop {d_f:.3f} vs random {d_r:.3f}', flush=True)
