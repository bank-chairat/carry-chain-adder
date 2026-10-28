# C-128: recompute the numbers quoted for C-121 to C-127 in ANSWERS.md from the raw files.
import csv, json, numpy as np
from scipy.stats import spearmanr
R = list(csv.DictReader(open('c121_sep/layer3.csv'))) + list(csv.DictReader(open('c121_plain/layer3.csv'))); v = lambda t: 6000 if int(t) < 0 else int(t)
for mode, arm in (('sep', 'none'), ('sep', 'L3'), ('plain', 'joint'), ('plain', 'blockL3')):
    rr = [r for r in R if r['mode'] == mode and r['arm'] == arm]
    print('C-121', mode, arm, 'median t4', np.median([v(r['t4']) for r in rr]), 'coh L2 %.2f relchange L2 %.2f' % (np.median([float(r['coh_oth_L2']) for r in rr]), np.median([float(r['relchange_L2_500']) for r in rr])))
W = list(csv.DictReader(open('c122/widthdup.csv'))); jm = {a: np.median([int(r['t4']) for r in W if r['arm'] == a and r['mode'] == 'joint']) for a in ('W128dup4', 'DUP512')}
for a, m in jm.items(): print('C-122', a, 'joint median', m, 'share of log width effect %.2f' % (np.log(2550 / m) / np.log(2550 / 500)))
b = list(csv.DictReader(open('c123b_race.csv'))); t = [int(r['time']) for r in b]
print('C-123b Spearman corr_125 %.2f corr_500 %.2f' % (spearmanr([float(r['corr_125']) for r in b], t).statistic, spearmanr([float(r['corr_500']) for r in b], t).statistic))
a = list(csv.DictReader(open('c123_race.csv'))); print('C-123 alone BCE at 100 Spearman %.2f' % spearmanr([float(r['alone_bce_100']) for r in a], [int(r['time']) for r in a]).statistic)
print(open('c123c_scores.txt').read().strip().splitlines()[3], '|', open('c123c_scores.txt').read().strip().splitlines()[-2])
on = json.load(open('c125_onsets.json'))
rates = []
for s in range(40):
    z = np.load(f'c125/traj_s{s}.npz'); st, c = z['steps'], z['corr']; k0 = list(st).index(on[s]['t_on'])
    idx = []
    for k in range(k0, len(c)):
        if 0.05 <= c[k] <= 0.3: idx.append(k)
        elif idx: break
    if len(idx) >= 3: rates.append(np.polyfit(st[idx].astype(float), np.log(c[idx]), 1)[0])
rates = np.array(rates); print('C-125 growth rates: n %d, min %.4f, 10th pct %.4f, median %.4f, 90th pct %.4f, max %.4f, 90/10 ratio %.1f' % (len(rates), rates.min(), np.percentile(rates, 10), np.median(rates), np.percentile(rates, 90), rates.max(), np.percentile(rates, 90) / np.percentile(rates, 10)))
print(open('c126_scores.txt').read().strip().splitlines()[-3]); print(open('c126b_scores.txt').read().strip())
print(open('c127_scores.txt').read().strip().splitlines()[-1])
