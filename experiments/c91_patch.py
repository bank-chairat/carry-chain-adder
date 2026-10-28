# NOTE: the first sampling loop inside the per-bit section is a leftover (it only advances the random stream prng); the results of C-91 depend on it, so it was not removed after the run.
import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); os.makedirs("c91_nets", exist_ok=True); n = 8; EV = 100; BS = 256; CAP = 12000
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1); Xall = torch.tensor(X)
def mknet():
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
def select(net):
    W1 = net[0].weight.detach().numpy(); W2 = net[2].weight.detach().numpy(); out = np.linalg.norm(W2, axis=0); keep = np.where(out >= np.median(out))[0]; G = []
    for u in keep:
        w = (np.abs(W1[u, :8]) + np.abs(W1[u, 8:])) / 2; m = w.max(); act = np.where(w >= 0.1 * m)[0]
        if m > 0 and len(act) >= 3 and np.polyfit(act, np.log2(w[act]), 1)[0] >= 0.5: G.append(int(u))
    return keep, G, out
def run_from_h1(net, h):
    return net[6](net[5](net[4](net[3](net[2](h)))))
w = csv.writer(open(f"{OUT}/patch_n8.csv", "w", newline="")); w.writerow(["seed", "bit", "set", "nG", "kept", "flip"])
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
for seed in SEEDS:
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    net = mknet(); path = f"c91_nets/net_s{seed}.pt"
    if os.path.exists(path): net.load_state_dict(torch.load(path, weights_only=True))
    else:
        Xtr, Ytr, Yte = torch.tensor(X[tr]), torch.tensor(S[tr]), torch.tensor(S[te])
        torch.manual_seed(seed * 100 + 4); rng = np.random.default_rng(seed * 100 + 4); net = mknet(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for step in range(1, CAP + 1):
            i = rng.integers(0, len(tr), BS)
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i], reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xall[te]) > 0).float() == Yte).float().mean(0).numpy()
                if (acc >= 0.99).all(): break
        torch.save(net.state_dict(), path)
    keep, G, outn = select(net); nG = len(G); rest = [int(u) for u in keep if int(u) not in G]; T = sorted(rest, key=lambda u: -outn[u])[:nG]
    prng = np.random.default_rng(1000 + seed)
    for bit in (3, 4, 5):
        low = (1 << bit) - 1; xs, xps = [], []
        cand = prng.choice(te, 4000, replace=False)
        for idx in cand:
            a, b = int(A[idx]), int(B[idx]); c = ((a & low) + (b & low)) >> bit
            for _ in range(200):
                a2 = (a & ~low) | int(prng.integers(0, 1 << bit)); b2 = (b & ~low) | int(prng.integers(0, 1 << bit)); c2 = ((a2 & low) + (b2 & low)) >> bit
                if c2 != c: xs.append(a * 256 + b if False else idx); xps.append(int(b2) * 256 + int(a2) if False else None); break
            else: continue
        # rebuild x' indices: A, B come from meshgrid(arange, arange).ravel(): index = b_row*256 + a_col ... use lookup
        lookup = {(int(A[k]), int(B[k])): k for k in range(0, len(A))}
        xs2, xps2 = [], []
        for idx in cand:
            a, b = int(A[idx]), int(B[idx]); c = ((a & low) + (b & low)) >> bit
            for _ in range(200):
                a2 = (a & ~low) | int(prng.integers(0, 1 << bit)); b2 = (b & ~low) | int(prng.integers(0, 1 << bit)); c2 = ((a2 & low) + (b2 & low)) >> bit
                if c2 != c: xs2.append(idx); xps2.append(lookup[(a2, b2)]); break
        xs2 = np.array(xs2); xps2 = np.array(xps2)
        with torch.no_grad():
            h = net[1](net[0](Xall[xs2])); hp = net[1](net[0](Xall[xps2])); clean = (run_from_h1(net, h)[:, bit] > 0).numpy(); cleanp = (run_from_h1(net, hp)[:, bit] > 0).numpy()
            truth = S[xs2, bit] > 0.5; truthp = S[xps2, bit] > 0.5; ok = (clean == truth) & (cleanp == truthp) & (truth != truthp)
            def flip(units):
                hh = h.clone(); hh[:, units] = hp[:, units]; out = (run_from_h1(net, hh)[:, bit] > 0).numpy()
                return float((out[ok] == cleanp[ok]).mean())
            rr = np.random.default_rng(seed); fr = np.mean([flip(list(rr.choice(rest, nG, replace=False))) for _ in range(10)])
            for name, v in (("G", flip(G)), ("T", flip(T)), ("R", fr), ("ALL", flip(list(range(128))))): w.writerow([seed, bit, name, nG, int(ok.sum()), round(v, 4)])
    print(seed, "nG", nG, flush=True)
