# C-86: fit T = F + Lambda/(lr*L1) on the 9 cells of C-39 and evaluate on data not in the fit. L1 values from C-84 / C-86 (c86_l1.json).
import numpy as np
L1 = {64: 27.1, 128: 63.3, 256: 163.7, 512: 422.5, 1024: 1179.4}
cells = [(128, 1e-3, 4000), (128, 3e-3, 1600), (128, 1e-2, 600), (256, 1e-3, 1500), (256, 3e-3, 700), (256, 1e-2, 400), (512, 1e-3, 700), (512, 3e-3, 400), (512, 1e-2, 300)]
x = np.array([1 / (lr * L1[W]) for W, lr, _ in cells]); T = np.array([c[2] for c in cells], float)
Lam, F = np.polyfit(x, T, 1); f = lambda W, lr: F + Lam / (lr * L1[W])
print("Lambda %.1f F %.1f" % (Lam, F)); print("in-sample pred/meas", [round(float(p / t), 2) for p, t in zip(F + Lam * x, T)])
print("W=64 lr 1e-3:", round(f(64, 1e-3)), "(measured >= 10000); W=1024 lr 1e-3:", round(f(1024, 1e-3)), "(500); lr 3e-4:", round(f(1024, 3e-4)), "(800)")
L1c = {1: 63.4, 2: 458.7, 4: 3567.8}; g = lambda c: F + Lam / (1e-3 * L1c[c])
for c, m in ((2, 550 / 2450), (4, 750 / 2450)): print("c=%d formula ratio %.3f measured %.3f" % (c, g(c) / g(1), m))
