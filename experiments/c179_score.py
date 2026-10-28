"""C-179 scoring: S-dominance of the last hidden layer at step 250 against the measured joint/alone ratio (C-141 list) over eight settings."""
import csv, numpy as np
from scipy.stats import spearmanr
ratio = {"W64": 33.3, "W128": 10.2, "W512": 3.3, "FW1024": 33.0, "CS2": 2.7, "CS4": 1.8, "lr3e-4": 22.9, "lr1e-2": 3.7}
P, Q, Z = [], [], []
for k, r in ratio.items():
    R = list(csv.DictReader(open(f"c179_{k}/confine.csv"))); p = np.median([float(x["L3_R2S250"]) for x in R]); q = np.median([float(x["L3norm_corrS250"]) for x in R]); z = np.median([float(x["L2_R2S250"]) for x in R])
    P.append(p); Q.append(q); Z.append(z); print(f"{k:7s} ratio {r:5.1f} | last-layer R2 on S {p:.3f} | last-layer norm-S corr {q:+.3f} | second-to-last R2 on S {z:.3f}")
rv = list(ratio.values())
print(f"Spearman(primary: last-layer R2 on S, ratio) {spearmanr(P, rv).statistic:+.2f} -> prediction (at least +0.6): {'HIT' if spearmanr(P, rv).statistic >= 0.6 else 'MISSED'}")
print(f"secondary: Spearman(last-layer norm-S corr, ratio) {spearmanr(Q, rv).statistic:+.2f}; second-to-last R2 {spearmanr(Z, rv).statistic:+.2f}")
