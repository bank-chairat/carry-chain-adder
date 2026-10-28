"""C-162 scoring: on the new full runs (s = 100-299), Spearman of the head start J(t) with the bit-4 time T, and a prospective test of the frozen onset law T_hat = 1,283 + 2.53 t_on."""
import numpy as np, glob, re
from scipy.stats import spearmanr
def onset(st, c):  # as c131_score.py
    for k in range(len(c) - 2):
        if c[k] >= 0.05 and c[k + 1] >= 0.05 and c[k + 2] >= 0.05: return int(st[k])
    return None
R = {}
for f in glob.glob("c162_*/traj_s*.npz"):
    z = np.load(f); s = int(re.findall(r"traj_s(\d+)", f)[0]); R[s] = z
S = sorted(R); T = np.array([int(R[s]["t4"]) if int(R[s]["t4"]) > 0 else 9000 for s in S]); cens = sum(int(R[s]["t4"]) < 0 for s in S)
print(f"{len(S)} runs (s {min(S)}-{max(S)}), censored at 9,000: {cens}; T median {np.median(T):.0f}, range {T.min()}-{T.max()}")
rho = {}
for t in (25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500):
    J = np.array([R[s]["corr"][list(R[s]["steps"]).index(t)] if t in list(R[s]["steps"]) else R[s]["corr"][-1] for s in S]); rho[t] = spearmanr(J, T).statistic
    print(f"t = {t:3d}: Spearman(J(t), T) {rho[t]:+.2f}")
print(f"(1) Spearman at 25 within +-0.15: {'HIT' if abs(rho[25]) <= 0.15 else 'MISSED'} ({rho[25]:+.2f})")
print(f"(2) Spearman at 125 at most -0.4: {'HIT' if rho[125] <= -0.4 else 'MISSED'} ({rho[125]:+.2f})")
print(f"(3) Spearman at 250 at most -0.55: {'HIT' if rho[250] <= -0.55 else 'MISSED'} ({rho[250]:+.2f})")
ok = [s for s in S if int(R[s]["t4"]) > 0 and onset(R[s]["steps"], R[s]["corr"]) is not None]
th = np.array([1283 + 2.53 * onset(R[s]["steps"], R[s]["corr"]) for s in ok]); tt = np.array([int(R[s]["t4"]) for s in ok])
err = np.abs(np.log(th / tt)); const = np.abs(np.log(2900 / tt))  # constant prediction = C-83 median, as in C-127
print(f"onset law on {len(ok)} uncensored runs with an onset: median |ln(T_hat/T)| {np.median(err):.2f} (max {err.max():.2f}); constant prediction 2,900 (C-83 median) {np.median(const):.2f}; Spearman(T_hat, T) {spearmanr(th, tt).statistic:.2f}")
print(f"(4) median |ln error| at most 0.25: {'HIT' if np.median(err) <= 0.25 else 'MISSED'}; (5) Spearman at least 0.8: {'HIT' if spearmanr(th, tt).statistic >= 0.8 else 'MISSED'}")
