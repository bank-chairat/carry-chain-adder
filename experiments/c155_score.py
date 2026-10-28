import glob, numpy as np
from scipy.stats import spearmanr
def onset(st, c):
    for k in range(len(c) - 2):
        if c[k] >= 0.05 and c[k + 1] >= 0.05 and c[k + 2] >= 0.05: return int(st[k])
    return None
def thalf(st, P):
    k0 = int(np.argmax(P)); pk = P[k0]
    for k in range(k0, len(P)):
        if P[k] < 0.5 * pk: return int(st[k])
    return None
rows = []
for f in sorted(glob.glob('c155_inits/*.npz')) + sorted(glob.glob('c155_settings/*.npz')):
    z = np.load(f); st, P, J, T = z['steps'], z['P'], z['J'], int(z['t4']); name = f.split('push_')[1].rsplit('_s', 1)[0]
    rows.append((name, f, T, onset(st, J), thalf(st, P), int(st[int(np.argmax(P))]), float(P.max())))
ini = [r for r in rows if r[0] == 'C83' and r[3] is not None and r[4] is not None]
print('inits with onset and t_half: %d of 40' % len(ini))
print('(1) Spearman(t_half, t_on) over inits %.2f; Spearman(t_half, T) %.2f; t_half median %.0f range %.0f-%.0f; peak step median %.0f' % (spearmanr([r[4] for r in ini], [r[3] for r in ini]).statistic, spearmanr([r[4] for r in ini], [r[2] for r in ini]).statistic, np.median([r[4] for r in ini]), min(r[4] for r in ini), max(r[4] for r in ini), np.median([r[5] for r in ini])))
allr = [r for r in rows if r[3] is not None and r[4] is not None]
print('(2) runs with t_on >= t_half - 100: %d of %d (%.0f%%)' % (sum(r[3] >= r[4] - 100 for r in allr), len(allr), 100 * np.mean([r[3] >= r[4] - 100 for r in allr])))
med = {}
for name in ('lr3e-4', 'lr1e-2', 'W64', 'W512', 'C83'):
    rr = [r for r in allr if r[0] == name]; med[name] = (np.median([r[4] for r in rr]), np.median([r[3] for r in rr]), np.median([r[2] if r[2] > 0 else 10000 for r in rr]))
    print(f'  {name:7s} t_half median {med[name][0]:.0f}  t_on median {med[name][1]:.0f}  T median {med[name][2]:.0f}  (n {len(rr)})')
ks = list(med); print('(3) Spearman over settings of medians (t_half, t_on): %.2f' % spearmanr([med[k][0] for k in ks], [med[k][1] for k in ks]).statistic)
