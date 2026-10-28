import json, numpy as np
from scipy.stats import spearmanr
on = {r['s']: r for r in json.load(open('c125_onsets.json'))}
rec = {}
for s in range(40):
    z = np.load(f'c125/traj_s{s}.npz'); st, c, T = z['steps'], z['corr'], int(z['t4']); t_on = on[s]['t_on']; k0 = list(st).index(t_on)
    seg = slice(k0, k0 + 5); x = st[seg].astype(float); y = np.log(np.clip(c[seg], 1e-4, None)); lam = np.polyfit(x, y, 1)[0]
    rho_T = float(c[list(st).index(T)]) if T in list(st) else float(c[-1])
    rec[s] = dict(T=T, t_on=t_on, rho_on=float(c[k0]), lam=lam, rho_T=rho_T)
rho_star = float(np.median([rec[s]['rho_T'] for s in range(0, 40, 2)]))
print('rho* (median correlation at confirmation, even inits) = %.3f; spread on even inits %.3f-%.3f' % (rho_star, min(rec[s]['rho_T'] for s in range(0, 40, 2)), max(rec[s]['rho_T'] for s in range(0, 40, 2))))
pred, true = [], []
for s in range(1, 40, 2):
    r = rec[s]; lam = r['lam']
    Th = r['t_on'] + np.log(rho_star / r['rho_on']) / lam if lam > 0 else np.inf
    pred.append(Th); true.append(r['T']); print('  s %2d T %5d T_hat %7.0f  t_on %4d lambda %.4f' % (s, r['T'], Th, r['t_on'], lam))
pred, true = np.array(pred), np.array(true)
fin = np.isfinite(pred)
print('held-out: Spearman %.2f; median |ln(T_hat/T)| %.2f (finite predictions %d of 20)' % (spearmanr(np.where(fin, pred, 1e9), true).statistic, np.median(np.abs(np.log(pred[fin] / true[fin]))), fin.sum()))
print('baseline without growth (T_hat = median T of even inits): median |ln| %.2f' % np.median(np.abs(np.log(np.median([rec[s]['T'] for s in range(0, 40, 2)]) / true))))
print('t_on alone (rank model): Spearman(t_on, T) on held-out %.2f' % spearmanr([rec[s]['t_on'] for s in range(1, 40, 2)], true).statistic)
