import numpy as np
from scipy.stats import spearmanr
def load(d, s):
    z = np.load(f"{d}/traj_s{s}.npz"); st = list(z["steps"]); return float(z["corr"][st.index(125)]), int(z["t4"])
fit = [load("c125", s) for s in range(40)]; test = [load("c127", s) for s in range(40, 60)]
J = np.array([j for j, _ in fit]); T = np.array([t for _, t in fit], float); b, a = np.polyfit(J, np.log(T), 1)
print("fitted: ln T = %.3f + %.3f J125  (T = %.0f at J = 0, %.0f at J = 0.1, %.0f at J = 0.2)" % (a, b, np.exp(a), np.exp(a + 0.1 * b), np.exp(a + 0.2 * b)))
Jt = np.array([j for j, _ in test]); Tt = np.array([t for _, t in test], float); pred = np.exp(a + b * Jt)
print("held-out: median |ln(T_hat/T)| %.2f (max %.2f); Spearman(J125, T) %.2f; J125 range %.3f to %.3f" % (np.median(np.abs(np.log(pred / Tt))), np.max(np.abs(np.log(pred / Tt))), spearmanr(Jt, Tt).statistic, Jt.min(), Jt.max()))
print("in-sample: median |ln| %.2f; Spearman %.2f" % (np.median(np.abs(np.log(np.exp(a + b * J) / T))), spearmanr(J, T).statistic))
