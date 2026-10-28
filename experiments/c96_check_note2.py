# C-96a: my own checks of claims in the second returned math note (nothing from their files is run)
import csv, numpy as np, statistics as st
from math import log2, sqrt
r = list(csv.DictReader(open("c44/aux4_n8.csv")))
for cond in ("p4", "c4", "p4c4"):
    x = [q for q in r if q["cond"] == cond]
    print(cond, "bit3", [int(q["t3"]) for q in x], "median", st.median(int(q["t3"]) for q in x), "| bit4", [int(q["t4"]) for q in x], "median", st.median(int(q["t4"]) for q in x))
p = {int(q["seed"]): q for q in r if q["cond"] == "p4"}; c = {int(q["seed"]): q for q in r if q["cond"] == "c4"}
print("c4 faster than p4 on bit 3 in", sum(int(c[s]["t3"]) < int(p[s]["t3"]) for s in p), "of", len(p), "seeds; p4 faster than c4 on bit 4 in", sum(int(p[s]["t4"]) < int(c[s]["t4"]) for s in p), "of", len(p))
idx = np.arange(1 << 16); a = idx & 255; b = idx >> 8; S = a + b; sp = lambda v: 1 - 2 * v
F = [sp((S >> i) & 1) for i in range(9)]; Q4 = sp(((a ^ b) >> 4) & 1); H4 = sp((((a & 15) + (b & 15)) >> 4) & 1)
print("E[Q4 F_j], j=0..8:", [round(float((Q4 * F[j]).mean()), 6) for j in range(9)], "(note: 1/16 = 0.0625, -1/256 = %.6f)" % (-1 / 256))
print("E[H4 F_j], j=0..8:", [round(float((H4 * F[j]).mean()), 6) for j in range(9)], "(note: -1/16 -1/8 -1/4 -1/2 = -0.0625 -0.125 -0.25 -0.5; 1/16 for F8)")
# eigenvalue algebra
for Qv, av in ((0.0, 1.0), (1.0, 1.0), (3.3222, 1.0), (10.0, 1.0)):
    lam = (sqrt(Qv ** 2 + 4 * av ** 2) - Qv) / 2; ev = np.linalg.eigvals(np.array([[-Qv, av], [av, 0.0]])).real.max(); print("Q=%.4f a=%.1f lambda formula %.5f numeric eig %.5f a/lambda %.4f" % (Qv, av, lam, ev, av / lam))
print("R=3.6 -> Q/a = %.5f" % (3.6 - 1 / 3.6))
# Hessian of the toy BCE realization at zero by finite differences
sp_ = lambda x: np.log1p(np.exp(x)); bb = 2.0
Lf = lambda u, v: sp_(-u * v) + 0.5 * (sp_(bb * u) + sp_(-bb * u)); h = 1e-4
H = np.array([[(Lf(h, 0) - 2 * Lf(0, 0) + Lf(-h, 0)) / h ** 2, (Lf(h, h) - Lf(h, -h) - Lf(-h, h) + Lf(-h, -h)) / (4 * h ** 2)], [0, (Lf(0, h) - 2 * Lf(0, 0) + Lf(0, -h)) / h ** 2]]); H[1, 0] = H[0, 1]
print("toy Hessian (b=2):", np.round(H, 4).tolist(), "formula [[b^2/4, -1/2],[-1/2, 0]] =", [[bb ** 2 / 4, -0.5], [-0.5, 0]])
# truncated carry rule: carry into bit i computed from only the last k positions, assuming zero carry into position i-k; bit i local parity right
for i in (4, 7):
    for k in range(0, i + 1):
        lo = i - k; m = (1 << i) - 1
        # carry into bit i from positions lo..i-1 with zero carry into lo
        seg = ((a >> lo) & ((1 << k) - 1)) + ((b >> lo) & ((1 << k) - 1)); c_hat = (seg >> k) & 1 if k > 0 else np.zeros(len(a), int)
        c_true = ((a & m) + (b & m)) >> i
        err = float((c_hat != c_true).mean()); formula = 2.0 ** -(k + 1) - 2.0 ** -(i + 1)
        if k in (0, i - 1, i): print("bit %d k=%d error %.6f formula %.6f" % (i, k, err, formula))
h2 = lambda p_: -(p_ * log2(p_) + (1 - p_) * log2(1 - p_)); print("1 - h2(15/32) = %.8f bits" % (1 - h2(15 / 32)))
