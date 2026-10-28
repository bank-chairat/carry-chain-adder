# C-137: recompute the numbers quoted in ANSWERS.md for C-129 to C-136b from the raw files.
import csv, numpy as np
med = np.median
r = list(csv.DictReader(open('c129/l3dil.csv'))); v = lambda t: 6000 if int(t) < 0 else int(t)
print('C-129', {a: med([v(x['t4']) for x in r if x['arm'] == a]) for a in ('joint', 'blockL3', 'blockL3_dilV')})
r = list(csv.DictReader(open('c129b/sepblock.csv'))); print('C-129b FO 1/8', {a: med([int(x['t4']) for x in r if x['blocked_layers'] == a]) for a in ('none', 'L3')})
for fo in ('0.25', '0.5', '2'):
    r = list(csv.DictReader(open(f'c132_FO{fo}/sepblock.csv'))); print('C-132 FO', fo, 'median', med([int(x['t4']) for x in r]), [int(x['t4']) for x in r])
print(open('c131_scores.txt').read().strip().splitlines()[-2])
print(open('c131_scores.txt').read().strip().splitlines()[-1])
print('C-130', open('c130_scores.txt').read().strip().splitlines()[1:6])
print(open('c133_scores.txt').read().strip())
C83 = {35: 6000, 5: 5800, 37: 5750, 36: 4300, 10: 4350, 25: 4500, 24: 5050, 38: 900, 23: 1200, 28: 1200}
for d in ('c135', 'c135ctl', 'c135c'):
    r = list(csv.DictReader(open(f'{d}/scale.csv')))
    for tag in sorted(set(x['tag'] for x in r)):
        rr = [x for x in r if x['tag'] == tag]; fast = all(int(x['s']) in (38, 23, 28) for x in rr)
        ratio = [int(x['t4']) / C83[int(x['s'])] if fast else C83[int(x['s'])] / int(x['t4']) for x in rr]
        print(d, tag, 'n', len(rr), ('slow-down' if fast else 'speed-up'), 'median %.2f' % med(ratio), 'J125 range %.3f..%.3f' % (min(float(x['corr125']) for x in rr), max(float(x['corr125']) for x in rr)))
print(open('c134_scores.txt').read().strip().splitlines()[1])
