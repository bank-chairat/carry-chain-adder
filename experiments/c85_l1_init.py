# C-85: gradient L1 norm and logit std of f_4 at init for the 40 inits of C-83, versus the joint bit-4 time (needs c83_a/, c83_b/ results)
import numpy as np, torch, torch.nn as nn, csv, json
torch.set_num_threads(1); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = torch.tensor(np.concatenate([bits(A, n), bits(B, n)], 1)); idx = np.random.default_rng(5).choice(len(X), 256, replace=False)
r = []
for d in ("c83_a", "c83_b"): r += list(csv.DictReader(open(f"{d}/init_align_joint_n8.csv")))
T = {int(x["s"]): int(x["time"]) for x in r}; rows = []
for s in range(40):
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, 9)); ps = list(net.parameters()); tot = 0.0
    for i in idx:
        g = torch.autograd.grad(net(X[i:i + 1])[0, 4], ps); tot += sum(float(x.abs().sum()) for x in g)
    with torch.no_grad(): sd = float(net(X[idx])[:, 4].std())
    rows.append((s, T[s], tot / len(idx), sd))
json.dump(rows, open("c85_rows.json", "w"))
