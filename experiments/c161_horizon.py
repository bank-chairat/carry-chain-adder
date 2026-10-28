"""C-161: AUC of J(t) for escape-by-500 (J500 >= 0.2 vs < 0.05) over the 1,000 saved initialisations, and Spearman(J(t), T) on the 60 with known T."""
import numpy as np, glob, csv, re
from scipy.stats import spearmanr, rankdata
Z = {int(re.findall(r"s(\d+)", f)[0]): np.load(f)["J"] for f in glob.glob("c139/init_s*.npz")}
steps = np.arange(0, 525, 25); S = sorted(Z); J = np.array([Z[s] for s in S])
assert J.shape[1] == len(steps)
lab = np.where(J[:, -1] >= 0.2, 1, np.where(J[:, -1] < 0.05, 0, -1)); keep = lab >= 0
def auc(x, y):
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
T = {}
for d in ("c83_a", "c83_b"):
    for r in csv.DictReader(open(f"{d}/init_align_joint_n8.csv")): T[int(r["s"])] = int(r["time"])
for s in range(40, 60): T[s] = int(np.load(f"c127/traj_s{s}.npz")["t4"])
k = [i for i, s in enumerate(S) if s in T]; Tk = [T[S[i]] for i in k]
print(f"{len(S)} initialisations, labelled {keep.sum()} (escaped {int((lab == 1).sum())}); {len(k)} with known T")
first = None
for j, t in enumerate(steps):
    if t == 0 or t == 500: continue
    a = auc(J[keep, j], lab[keep]); rho = spearmanr(J[k, j], Tk).statistic
    if first is None and a >= 0.8: first = t
    print(f"t = {t:3d}: AUC {a:.3f}   Spearman(J(t), T) on known-T inits {rho:+.2f}")
a50 = auc(J[keep, 2], lab[keep]); a250 = auc(J[keep, 10], lab[keep])
print(f"(1) AUC at 50 <= 0.65: {'HIT' if a50 <= 0.65 else 'MISSED'} ({a50:.3f})")
print(f"(2) AUC at 250 >= 0.85: {'HIT' if a250 >= 0.85 else 'MISSED'} ({a250:.3f})")
print(f"(3) first t with AUC >= 0.8 in 100-300: {'HIT' if first is not None and 100 <= first <= 300 else 'MISSED'} ({first})")
