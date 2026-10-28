import sys, os, csv, numpy as np, torch, torch.nn as nn
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); n = 8; SEEDS = [int(x) for x in os.environ.get("SEEDS", "5 6 7 8 9").split()]
def mknet():
    L, d = [], 2 * n
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    return nn.Sequential(*L, nn.Linear(d, n + 1))
def select(net):
    W1 = net[0].weight.detach().numpy(); W2 = net[2].weight.detach().numpy(); out = np.linalg.norm(W2, axis=0); keep = np.where(out >= np.median(out))[0]; G = []
    for u in keep:
        w = (np.abs(W1[u, :8]) + np.abs(W1[u, 8:])) / 2; m = w.max(); act = np.where(w >= 0.1 * m)[0]
        if m > 0 and len(act) >= 3 and np.polyfit(act, np.log2(w[act]), 1)[0] >= 0.5: G.append(int(u))
    return keep, G
w = csv.writer(open(f"{OUT}/antisym_n8.csv", "w", newline="")); w.writerow(["seed", "set", "unit", "k", "r"])
for seed in SEEDS:
    net = mknet(); net.load_state_dict(torch.load(f"c91_nets/net_s{seed}.pt", weights_only=True)); keep, G = select(net); rest = [int(u) for u in keep if int(u) not in G]
    R = list(np.random.default_rng(seed).choice(rest, len(G), replace=False)); W1 = net[0].weight.detach().numpy()
    for name, units in (("G", G), ("R", R)):
        for u in units:
            wa, wb = W1[u, :8], W1[u, 8:]; k = int(np.argmax(np.abs(wa) + np.abs(wb))); r = abs(wa[k] + wb[k]) / (abs(wa[k]) + abs(wb[k]) + 1e-12); w.writerow([seed, name, int(u), k, round(float(r), 4)])
