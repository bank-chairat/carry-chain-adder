import numpy as np, torch, csv, scipy.sparse.linalg as sla
torch.set_num_threads(4); torch.set_default_dtype(torch.float64); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float64)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
shapes = [(128, 16), (128,), (128, 128), (128,), (128, 128), (128,), (9, 128), (9,)]
sizes = [int(np.prod(s)) for s in shapes]; NB = sum(sizes[:6])   # body = first six tensors
rows = []
for seed in range(3):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr = perm[:len(A) // 2]
    idx = tr[np.random.default_rng(7).choice(len(tr), 1024, replace=False)]
    Xd, Yd = torch.tensor(X[idx]), torch.tensor(Y[idx])
    for step in (50, 100, 200, 300):
        ck = torch.load(f"c102_ckpt/s{seed}_t{step}.pt", weights_only=True)
        sd = ck["net"]; names = ["0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias", "6.weight", "6.bias"]
        full = torch.cat([sd[k].double().reshape(-1) for k in names])
        # coordinates: body + head row 4 (weight row, bias)
        hw = 6 * 0 + NB; head_w = full[NB:NB + 9 * 128].view(9, 128); head_b = full[NB + 9 * 128:]
        w0 = torch.cat([full[:NB], head_w[4], head_b[4:5]])
        # Adam second moment, same coordinates
        st = ck["opt"]["state"]; v = torch.cat([st[i]["exp_avg_sq"].double().reshape(-1) for i in range(8)]); t = int(st[0]["step"])
        vhat = v / (1 - 0.999 ** t); vh_head = vhat[NB:NB + 9 * 128].view(9, 128); vh_b = vhat[NB + 9 * 128:]
        v0 = torch.cat([vhat[:NB], vh_head[4], vh_b[4:5]]); d = 1.0 / (v0.sqrt() + 1e-8); dh = d.sqrt()
        def logits(w):
            p = []; o = 0
            for s, z in zip(shapes[:6], sizes[:6]): p.append(w[o:o + z].view(s)); o += z
            h = torch.relu(Xd @ p[0].T + p[1]); h = torch.relu(h @ p[2].T + p[3]); h = torch.relu(h @ p[4].T + p[5])
            return h, w[NB:NB + 128], w[NB + 128], head_w, head_b
        def zout(w):
            h, w4, b4, hw_, hb_ = logits(w)
            z = h @ hw_.T + hb_                    # other heads frozen at their saved values
            z4 = h @ w4 + b4
            return z, z4
        def L4(w): z, z4 = zout(w); return torch.nn.functional.binary_cross_entropy_with_logits(z4, Yd[:, 4])
        oth = [j for j in range(9) if j != 4]
        def Lo(w): z, _ = zout(w); return torch.nn.functional.binary_cross_entropy_with_logits(z[:, oth], Yd[:, oth], reduction="none").sum(1).mean()
        def hvp(L, w, u):
            w = w.clone().requires_grad_(True); g, = torch.autograd.grad(L(w), w, create_graph=True)
            return torch.autograd.grad((g * u).sum(), w)[0]
        N = w0.numel()
        op = sla.LinearOperator((N, N), matvec=lambda u: (dh * hvp(L4, w0, dh * torch.tensor(u))).numpy(), dtype=np.float64)
        ev, evec = sla.eigsh(op, k=1, which="SA", tol=1e-6, ncv=40)
        q = torch.tensor(evec[:, 0]); lam4 = float(ev[0])
        qS4 = float(q @ (dh * hvp(L4, w0, dh * q))); qSo = float(q @ (dh * hvp(Lo, w0, dh * q)))
        # Gauss-Newton part of q'S_o q: direction u = dh*q, forward derivative of logits
        u = dh * q
        jv = torch.func.jvp(lambda w: zout(w)[0], (w0,), (u,))[1]; z, _ = zout(w0); s = torch.sigmoid(z)
        gn = float(((s * (1 - s))[:, oth] * jv[:, oth] ** 2).sum(1).mean())
        # gradient cosine on these coordinates
        wr = w0.clone().requires_grad_(True); g4, = torch.autograd.grad(L4(wr), wr); go, = torch.autograd.grad(Lo(wr), wr)
        cos = float(g4 @ go / (g4.norm() * go.norm()))
        rnd = []
        gr = torch.Generator().manual_seed(11)
        for _ in range(20):
            r = torch.randn(N, generator=gr); r = r / r.norm(); rnd.append(float(r @ (dh * hvp(Lo, w0, dh * r))))
        rows.append(dict(seed=seed, step=step, lam_min_S4=lam4, qS4q=qS4, qSoq=qSo, joint=qS4 + qSo, gn=gn, gn_frac=gn / qSo if qSo else float('nan'), cos=cos, rand_mean=float(np.mean(rnd)), ratio=qSo / float(np.mean(rnd)) if np.mean(rnd) else float('nan')))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
with open("c102_qdir.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
