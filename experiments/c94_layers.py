import sys, os, csv, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel(); X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1); Xall = torch.tensor(X)
lookup = {(int(A[k]), int(B[k])): k for k in range(len(A))}
def mknet():
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
def acts(net, x):
    h1 = net[1](net[0](x)); h2 = net[3](net[2](h1)); h3 = net[5](net[4](h2)); return [h1, h2, h3]
def from_layer(net, L, h):
    if L == 0: h = net[3](net[2](h)); L = 1
    if L == 1: h = net[5](net[4](h)); L = 2
    return net[6](h)
def pairs(prng, te, bit, N):
    low = (1 << bit) - 1; xs, xps = [], []
    for idx in prng.choice(te, N, replace=False):
        a, b = int(A[idx]), int(B[idx]); c = ((a & low) + (b & low)) >> bit
        for _ in range(200):
            a2 = (a & ~low) | int(prng.integers(0, 1 << bit)); b2 = (b & ~low) | int(prng.integers(0, 1 << bit))
            if (((a2 & low) + (b2 & low)) >> bit) != c: xs.append(idx); xps.append(lookup[(a2, b2)]); break
    return np.array(xs), np.array(xps)
MS = [1, 2, 4, 8, 16, 32, 64]
w = csv.writer(open(f"{OUT}/layers_n8.csv", "w", newline="")); w.writerow(["seed", "bit", "layer", "m", "set", "flip"])
for seed in range(10):
    perm = np.random.default_rng(500 + seed).permutation(len(A)); te = perm[len(A) // 2:]
    net = mknet(); net.load_state_dict(torch.load(f"c91_nets/net_s{seed}.pt", weights_only=True)); prng = np.random.default_rng(2000 + seed)
    for bit in (4, 5):
        sel = pairs(prng, te, bit, 3000); ev = pairs(prng, te, bit, 3000)
        with torch.no_grad():
            def prep(pr):
                xs, xps = pr; H, Hp = acts(net, Xall[xs]), acts(net, Xall[xps])
                clean = (from_layer(net, 2, H[2])[:, bit] > 0).numpy(); cleanp = (from_layer(net, 2, Hp[2])[:, bit] > 0).numpy()
                ok = (clean == (S[xs, bit] > 0.5)) & (cleanp == (S[xps, bit] > 0.5)) & ((S[xs, bit] > 0.5) != (S[xps, bit] > 0.5)); return H, Hp, ok, cleanp
            Hs, Hps, oks, cps = prep(sel); He, Hpe, oke, cpe = prep(ev)
            for L in range(3):
                # single-unit effect on selection pairs: change of the bit logit in the direction of the counterfactual output
                sign = torch.tensor(np.where(cps[oks], 1.0, -1.0), dtype=torch.float32); eff = np.zeros(128)
                Hsk = Hs[L][torch.tensor(oks)]; Hpsk = Hps[L][torch.tensor(oks)]
                for u in range(128):
                    hh = Hsk.clone(); hh[:, u] = Hpsk[:, u]
                    out = from_layer(net, L, hh)
                    eff[u] = float((sign * out[:, bit]).mean())
                order = np.argsort(-eff); rr = np.random.default_rng(seed)
                for m in MS:
                    for name, units in (("top", list(order[:m])), ("rand", list(rr.choice(128, m, replace=False)))):
                        hh = He[L][torch.tensor(oke)].clone(); hh[:, units] = Hpe[L][torch.tensor(oke)][:, units]
                        o = (from_layer(net, L, hh)[:, bit] > 0).numpy(); w.writerow([seed, bit, L + 1, m, name, round(float((o == cpe[oke]).mean()), 4)])
    print(seed, flush=True)
