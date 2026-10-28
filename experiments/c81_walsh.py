import numpy as np, json
idx = np.arange(1 << 16); a = idx & 255; b = idx >> 8; S = a + b       # input index: bits 0-7 = a, bits 8-15 = b
def wht(f):
    f = f.astype(np.float64).copy(); h = 1
    while h < len(f):
        f = f.reshape(-1, 2, h); x = f[:, 0, :].copy(); y = f[:, 1, :].copy(); f[:, 0, :] = x + y; f[:, 1, :] = x - y; f = f.reshape(-1); h *= 2
    return f / len(f)
pc = np.array([bin(i).count("1") for i in range(1 << 16)])
for k in range(9):
    c = wht(1 - 2 * ((S >> k) & 1)); w = c ** 2; Wd = [float(w[pc == d].sum()) for d in range(17)]
    print(k, "W0-2 %.3f mean_deg %.2f max|c| %.3f nonzero %d" % (sum(Wd[:3]), sum(d * Wd[d] for d in range(17)), np.abs(c).max(), int((np.abs(c) > 1e-9).sum())), [round(x, 3) for x in Wd[:9]])
