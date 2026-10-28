import sys, os, csv, json, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); os.makedirs("c95_nets", exist_ok=True); n = 12; EV = 200; BS = 256; CAP = int(os.environ.get("CAP", "60000"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
erng = np.random.default_rng(70000 + n); ea = erng.integers(0, 1 << n, 8192); eb = erng.integers(0, 1 << n, 8192)
excl = np.unique(np.concatenate([(ea << n) | eb, (eb << n) | ea])); Xe = torch.tensor(np.concatenate([bits(ea, n), bits(eb, n)], 1)); Ye = bits(ea + eb, n + 1)
def mknet():
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
def select(net):
    W1 = net[0].weight.detach().numpy(); W2 = net[2].weight.detach().numpy(); out = np.linalg.norm(W2, axis=0); keep = np.where(out >= np.median(out))[0]; G = []
    for u in keep:
        w = (np.abs(W1[u, :n]) + np.abs(W1[u, n:])) / 2; m = w.max(); act = np.where(w >= 0.1 * m)[0]
        if m > 0 and len(act) >= 3 and np.polyfit(act, np.log2(w[act]), 1)[0] >= 0.5: G.append(int(u))
    return keep, G, out
def head(net, h): return net[6](net[5](net[4](net[3](net[2](h)))))
wa_ = csv.writer(open(f"{OUT}/ablate_n12.csv", "w", newline="")); wa_.writerow(["seed", "group", "nG"] + [f"drop{b}" for b in range(n + 1)])
wp_ = csv.writer(open(f"{OUT}/patch_n12.csv", "w", newline="")); wp_.writerow(["seed", "bit", "set", "nG", "kept", "flip"])
wr_ = csv.writer(open(f"{OUT}/antisym_n12.csv", "w", newline="")); wr_.writerow(["seed", "set", "unit", "k", "r"])
wt_ = csv.writer(open(f"{OUT}/train_n12.csv", "w", newline="")); wt_.writerow(["seed", "steps", "min_acc", "all_ok"])
for seed in SEEDS:
    net = mknet(); path = f"c95_nets/net_s{seed}.pt"
    if os.path.exists(path): net.load_state_dict(torch.load(path, weights_only=True)); steps, ok_all = -1, True
    else:
        t0 = time.time(); rng = np.random.default_rng(1000 + seed); torch.manual_seed(seed); net = mknet(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); prev_ok = False; ok_all = False
        for step in range(1, CAP + 1):
            a = rng.integers(0, 1 << n, BS * 2); b = rng.integers(0, 1 << n, BS * 2); ok = ~np.isin((a << n) | b, excl); a, b = a[ok][:BS], b[ok][:BS]
            x = torch.tensor(np.concatenate([bits(a, n), bits(b, n)], 1)); y = torch.tensor(bits(a + b, n + 1))
            loss = nn.functional.binary_cross_entropy_with_logits(net(x), y, reduction="none").sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step % EV == 0:
                with torch.no_grad(): acc = ((net(Xe) > 0).float().numpy() == Ye).mean(0)
                good = bool((acc >= 0.99).all())
                if good and prev_ok: ok_all = True; break
                prev_ok = good
        steps = step; torch.save(net.state_dict(), path)
    with torch.no_grad(): accf = ((net(Xe) > 0).float().numpy() == Ye).mean(0)
    wt_.writerow([seed, steps, round(float(accf.min()), 4), int(ok_all)]); print("trained", seed, steps, round(float(accf.min()), 4), flush=True)
    keep, G, outn = select(net); nG = len(G); rest = [int(u) for u in keep if int(u) not in G]; T = sorted(rest, key=lambda u: -outn[u])[:nG]
    rr = np.random.default_rng(seed); Rs = [list(rr.choice(rest, nG, replace=False)) for _ in range(10)] if nG <= len(rest) else []
    if nG == 0: continue
    with torch.no_grad(): mean1 = net[1](net[0](Xe)).mean(0)         # note: mean over the held-out panel
    def accs(units):
        with torch.no_grad():
            h = net[1](net[0](Xe)); h[:, units] = mean1[units] if len(units) else h[:, units]; return ((head(net, h) > 0).float().numpy() == Ye).mean(0)
    base = accs([]); wa_.writerow([seed, "G", nG] + [round(float(base[b] - a), 4) for b, a in enumerate(accs(G))]); wa_.writerow([seed, "T", nG] + [round(float(base[b] - a), 4) for b, a in enumerate(accs(T))])
    wa_.writerow([seed, "R", nG] + [round(float(x), 4) for x in np.mean([base - accs(R) for R in Rs], axis=0)])
    W1 = net[0].weight.detach().numpy(); Rset = Rs[0]
    for name, units in (("G", G), ("R", Rset)):
        for u in units:
            wa, wb = W1[u, :n], W1[u, n:]; k = int(np.argmax(np.abs(wa) + np.abs(wb))); wr_.writerow([seed, name, int(u), k, round(float(abs(wa[k] + wb[k]) / (abs(wa[k]) + abs(wb[k]) + 1e-12)), 4)])
    prng = np.random.default_rng(3000 + seed)
    for bit in (6, 7, 8):
        low = (1 << bit) - 1; xa, xb, xa2, xb2 = [], [], [], []
        for _ in range(3000):
            a = int(prng.integers(0, 1 << n)); b = int(prng.integers(0, 1 << n)); c = ((a & low) + (b & low)) >> bit
            for _ in range(200):
                a2 = (a & ~low) | int(prng.integers(0, 1 << bit)); b2 = (b & ~low) | int(prng.integers(0, 1 << bit))
                if (((a2 & low) + (b2 & low)) >> bit) != c: xa.append(a); xb.append(b); xa2.append(a2); xb2.append(b2); break
        xa, xb, xa2, xb2 = map(np.array, (xa, xb, xa2, xb2)); x = torch.tensor(np.concatenate([bits(xa, n), bits(xb, n)], 1)); xp = torch.tensor(np.concatenate([bits(xa2, n), bits(xb2, n)], 1))
        ty = bits(xa + xb, n + 1)[:, bit] > 0.5; typ = bits(xa2 + xb2, n + 1)[:, bit] > 0.5
        with torch.no_grad():
            h = net[1](net[0](x)); hp = net[1](net[0](xp)); cl = (head(net, h)[:, bit] > 0).numpy(); clp = (head(net, hp)[:, bit] > 0).numpy(); ok = (cl == ty) & (clp == typ) & (ty != typ)
            def flip(units):
                hh = h.clone(); hh[:, units] = hp[:, units]; o = (head(net, hh)[:, bit] > 0).numpy(); return float((o[ok] == clp[ok]).mean())
            for name, v in (("G", flip(G)), ("T", flip(T)), ("R", float(np.mean([flip(R) for R in Rs]))), ("ALL", flip(list(range(128))))): wp_.writerow([seed, bit, name, nG, int(ok.sum()), round(v, 4)])
    print("analysed", seed, "nG", nG, flush=True)
