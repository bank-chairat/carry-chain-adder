import sys, os, glob, csv, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); S = bits(A + B, n + 1)
rows = []
for f in sorted(glob.glob("c105_?/net_c4_*_t1000.pt")):
    b = os.path.basename(f); cond = b.split("_")[1]; seed = int(b.split("_s")[1].split("_")[0])
    perm = np.random.default_rng(500 + seed).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
    net = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10)); net.load_state_dict(torch.load(f, weights_only=True))
    with torch.no_grad():
        Ftr = net[:6](torch.tensor(X[tr])).numpy().astype(np.float64); Fte = net[:6](torch.tensor(X[te])).numpy().astype(np.float64)
        native = float(((net(torch.tensor(X[te]))[:, 4] > 0).float().numpy() == S[te][:, 4]).mean())
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    clf = LogisticRegression(C=1.0 / (len(tr) * 1e-6), max_iter=30000, tol=1e-10).fit((Ftr - mu) / sd, S[tr][:, 4])
    fitted = float((clf.predict((Fte - mu) / sd) == S[te][:, 4]).mean()); fitted_tr = float((clf.predict((Ftr - mu) / sd) == S[tr][:, 4]).mean())
    rows.append((cond, seed, native, fitted_tr, fitted)); print(cond, seed, "native %.4f  fitted(train) %.4f  fitted(test) %.4f" % (native, fitted_tr, fitted), flush=True)
with open("c105_readout_long.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cond", "seed", "native_test", "fitted_train", "fitted_test"]); w.writerows(rows)
