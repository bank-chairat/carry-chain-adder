import glob, numpy as np
def onset(st, c):
    for k in range(len(c) - 2):
        if c[k] >= 0.05 and c[k + 1] >= 0.05 and c[k + 2] >= 0.05: return int(st[k])
    return None
res = {}
for arm, d in (("W128", "c131_b"), ("DUP512", "c131_a"), ("W512", "c131_a")):
    rows = []
    for s in range(5):
        z = np.load(f"{d}/traj_{arm}_s{s}.npz"); T = int(z["t4"]); on = onset(z["steps"], z["corr"]); rows.append((s, T, on, T - on))
    res[arm] = rows
    T = [r[1] for r in rows]; on = [r[2] for r in rows]; D = [r[3] for r in rows]
    print(f"{arm:7s} T {T} t_on {on} D {D} | medians T {np.median(T):.0f} t_on {np.median(on):.0f} D {np.median(D):.0f}")
m = lambda a, i: np.median([r[i] for r in res[a]])
print("DUP512/W512 ratio of medians: onset %.2f, growth phase %.2f, total %.2f" % (m("DUP512", 2) / m("W512", 2), m("DUP512", 3) / m("W512", 3), m("DUP512", 1) / m("W512", 1)))
print("W128/DUP512 ratio of medians: onset %.2f, growth phase %.2f, total %.2f" % (m("W128", 2) / m("DUP512", 2), m("W128", 3) / m("DUP512", 3), m("W128", 1) / m("DUP512", 1)))
