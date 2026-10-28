# C-88: my own check of formulas in the returned math_derivations.md (own computation; nothing run from their files)
import numpy as np
from math import comb
idx = np.arange(1 << 16); a = idx & 255; b = idx >> 8; S = a + b
U = [1 - 2 * ((a >> i) & 1) for i in range(8)]; V = [1 - 2 * ((b >> i) & 1) for i in range(8)]
bit = lambda x, i: (x >> i) & 1
F = [1 - 2 * bit(S, i) for i in range(9)]
H = [None] * 9; H[0] = np.ones(len(idx))
for i in range(8): H[i + 1] = (U[i] + V[i]) / 2 + (1 - U[i] * V[i]) * H[i] / 2           # their eq. (2)
carry = lambda i: 1 - 2 * (((a & ((1 << i) - 1)) + (b & ((1 << i) - 1))) >> i & 1) if i > 0 else np.ones(len(idx))
print("eq (2) matches true carry spin for i=1..8:", all(bool((H[i] == carry(i)).all()) for i in range(1, 9)))
print("E[H_i] = 2^-i:", all(abs(H[i].mean() - 2.0 ** -i) < 1e-12 for i in range(9)))
print("F_8 = H_8:", bool((F[8] == H[8]).all()), "| F_i = Q_i H_i for i<8:", all(bool((F[i] == U[i] * V[i] * H[i]).all()) for i in range(8)))
print("E[F_8 F_i] = -2^(i-8) for i<8:", all(abs((F[8] * F[i]).mean() + 2.0 ** (i - 8)) < 1e-12 for i in range(8)), "| E[F_i F_j]=0 for i!=j<8:", all(abs((F[i] * F[j]).mean()) < 1e-12 for i in range(8) for j in range(i + 1, 8)))
print("eq (5) E[F_i Q_i U_(i-1)] = 1/2 for i=1..7:", all(abs((F[i] * U[i] * V[i] * U[i - 1]).mean() - 0.5) < 1e-12 for i in range(1, 8)))
def wht(f):
    f = f.astype(np.float64).copy(); h = 1
    while h < len(f):
        f = f.reshape(-1, 2, h); x = f[:, 0, :].copy(); y = f[:, 1, :].copy(); f[:, 0, :] = x + y; f[:, 1, :] = x - y; f = f.reshape(-1); h *= 2
    return f / len(f)
pc = np.array([bin(i).count("1") for i in range(1 << 16)])
for i in (2, 4, 7):
    w = wht(F[i]) ** 2; Wd = [float(w[pc == d].sum()) for d in range(17)]; ok = True
    for r in range(i + 1): ok &= abs(Wd[2 * r + 2] - comb(i, r) * 4.0 ** -i) < 1e-12
    for r in range(i): ok &= abs(Wd[2 * r + 3] - 2 * sum(comb(d - 1, r) * 4.0 ** -d for d in range(r + 1, i + 1))) < 1e-12
    print("bit %d: their closed forms for W_d match the exact spectrum:" % i, ok, "| W_3 = %.7f, (2/3)(1-4^-i) = %.7f" % (Wd[3], 2 / 3 * (1 - 4.0 ** -i)))
d, eta = 0.01, 0.001; L1 = lambda th: 0.5 * (d * th.sum() - 1) ** 2
print("toy Adam example: solo L1 = %.10f, joint L1 = %.10f" % (L1(np.array([eta, eta])), L1(np.array([eta, -eta]))))
a0 = 0.01; astar = 1.0; T = np.trapezoid((1 + np.exp(np.linspace(a0, astar, 200001) ** 2)) / np.linspace(a0, astar, 200001), np.linspace(a0, astar, 200001))
print("composition toy: passage time %.2f, bounds %.2f to %.2f" % (T, 2 * np.log(astar / a0), (1 + np.exp(astar ** 2)) * np.log(astar / a0)))
