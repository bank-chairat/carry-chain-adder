import csv, glob, numpy as np
def med(d):
    r = list(csv.DictReader(open(f'{d}/sepblock.csv'))); return np.median([int(x['t4']) if int(x['t4']) > 0 else 8000 for x in r]), [int(x['t4']) for x in r]
meas = {'lr3e-4': 22.9, 'lr1e-2': 3.7, 'frozen': 33.0}; share = {'lr3e-4': 0.193, 'lr1e-2': 0.206, 'frozen': 0.223}
pred = {}
for k in meas:
    t = {fo: med(f'c150_{k}_FO{fo}') for fo in ('0.125', '0.25', '0.5')}
    T1, T2, T4 = t['0.125'][0], t['0.25'][0], t['0.5'][0]
    M2, M4 = T2 / T1, T4 / T1
    s = share[k]; r = (1 - s) / s
    # log-log interpolation / extrapolation through (2, M2) and (4, M4); M = 1 below r = 1
    lm = np.interp(np.log(r), [np.log(2), np.log(4)], [np.log(max(M2, 1e-9)), np.log(M4)]) if r <= 4 else np.log(M4) + (np.log(r) - np.log(4)) * (np.log(M4) - np.log(M2)) / np.log(2)
    pred[k] = (1 / s) * np.exp(lm)
    print(f'{k:7s} T(r=1) {t["0.125"][1]} T(r=2) {t["0.25"][1]} T(r=4) {t["0.5"][1]} | M(2) {M2:.2f} M(4) {M4:.2f} | s {s} r {r:.1f} -> predicted ratio {pred[k]:.1f}, measured {meas[k]}, ln(pred/meas) {np.log(pred[k]/meas[k]):+.2f}')
print('ordering lr3e-4 > lr1e-2 predicted:', pred['lr3e-4'] > pred['lr1e-2'], '| median |ln(pred/meas)| %.2f' % np.median([abs(np.log(pred[k]/meas[k])) for k in meas]))
