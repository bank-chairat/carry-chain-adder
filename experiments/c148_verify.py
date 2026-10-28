# C-148: recompute the numbers quoted in ANSWERS.md for C-138 to C-147 from saved outputs.
import json, csv, numpy as np
from scipy.stats import spearmanr
print(open('c138_phases.txt').read().strip())
print(open('c140_scores.txt').read().strip())
print(open('c141_scores.txt').read().strip().splitlines()[-1])
print(open('c139_scores.txt').read().strip().splitlines()[0]); print(open('c139_scores.txt').read().strip().splitlines()[-2])
S = list(range(60)) + list(range(100, 500)); J = np.array([float(np.load(f'c139/init_s{s}.npz')['J'][-1]) for s in S])
print('C-139 J500: escaped (>=0.2) %d, none (<0.05) %d, between %d of %d' % ((J >= 0.2).sum(), (J < 0.05).sum(), ((J >= 0.05) & (J < 0.2)).sum(), len(J)))
print(open('c142_scores.txt').read().strip().splitlines()[-3]); print(open('c142_scores.txt').read().strip().splitlines()[-2])
print(open('c143_scores.txt').read().strip())
r = list(csv.DictReader(open('c145/seed.csv'))) + list(csv.DictReader(open('c145b/seed.csv'))); base = {28: 1200, 26: 1500, 38: 900, 8: 1900, 11: 1450, 22: 1750, 35: 6000, 5: 5800, 37: 5750, 36: 4300, 10: 4350, 25: 4500, 24: 5050}
for arm in ('remove_seed', 'remove_median', 'plant_seed', 'random_nudge'):
    rr = [x for x in r if x['arm'] == arm]; ratio = [int(x['t4']) / base[int(x['s'])] for x in rr]
    print('C-145', arm, 'median time ratio (arm/baseline) %.2f' % np.median(ratio))
print(open('c146_scores.txt').read().strip().splitlines()[0]); print(open('c147_scores.txt').read().strip().splitlines()[-2])
