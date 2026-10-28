# C-120: recompute every number quoted in ANSWERS.md's 'whys answered again' section from the raw CSVs.
import csv, numpy as np
from scipy.stats import spearmanr
def rows(*fs):
    out = []
    for f in fs: out += list(csv.DictReader(open(f)))
    return out
med = lambda x: float(np.median(x))
a = rows('c113a.csv'); print('C-113a L2 R2S all9 %.3f  L3 R2S all9 %.3f  L2 R2S init %.3f' % (med([float(r['L2_R2S']) for r in a if r['kind']=='all9']), med([float(r['L3_R2S']) for r in a if r['kind']=='all9']), med([float(r['L2_R2S']) for r in a if r['kind']=='init'])))
d = rows('c111_a/diag.csv', 'c111_b/diag.csv'); print('C-111 L2 norm ratio K100 %.2f K1000 %.2f' % (med([float(r['L2_normratio']) for r in d if r['cond']=='all9_K100']), med([float(r['L2_normratio']) for r in d if r['cond']=='all9_K1000'])))
g = rows('c112/norm_geom.csv'); none = {0: 250, 1: 200, 2: 300, 3: 250, 4: 300}; D = {(r['arm'], int(r['seed'])): int(r['cost']) for r in g}
for arm in ('N_L2', 'N_all', 'D_L2'): print('C-112', arm, 'median g %.2f' % med([(D[('base', s)] - D[(arm, s)]) / (D[('base', s)] - none[s]) for s in range(5)]))
c = rows('c113b_a/companions.csv', 'c113b_b/companions.csv')
for k in ('alone', 'upper4', 'Sthr4', 'Rthr4'): print('C-113b', k, 'median t4', med([int(r['t4']) for r in c if r['cond']==k]))
b = rows('c114b_a/sepblock.csv', 'c114b_b/sepblock.csv'); v = lambda t: 6000 if int(t) < 0 else int(t)
for k in ('none', 'L12', 'all', 'L3'): print('C-114b', k, 'median', med([v(r['t4']) for r in b if r['blocked_layers']==k]))
print('C-114b Spearman L2 R2S vs t4 %.2f' % spearmanr([float(r['L2_R2S250']) for r in b], [v(r['t4']) for r in b]).statistic)
for name, f in (('W64', 'c116_W64_FW0'), ('W128', 'c116_W128_FW0'), ('W512', 'c116_W512_FW0'), ('frozen', 'c116_W128_FW1024'), ('CS2', 'c116_W128_CS2'), ('CS4', 'c116_W128_CS4'), ('lr3e-4', 'c116_W128_lr3e-4'), ('lr1e-2', 'c116_W128_lr1e-2')):
    R = rows(f'{f}/confine.csv'); vv = lambda t: 10000 if int(t) < 0 else int(t)
    m = {k: med([vv(r['t4']) for r in R if r['arm']==k]) for k in ('alone', 'conf3', 'joint')}
    print('C-116', name, 'alone %.0f conf3 %.0f joint %.0f conf3/alone %.2f joint/alone %.1f joint L2 R2S %.2f' % (m['alone'], m['conf3'], m['joint'], m['conf3']/m['alone'], m['joint']/m['alone'], med([float(r['L2_R2S250']) for r in R if r['arm']=='joint'])))
s = rows('c119_a/l1parts.csv', 'c119_b/l1parts.csv'); fast = {('38','35'): 900, ('23','5'): 1200, ('28','37'): 1200}
singles = [int(r['t4']) for r in s if r['combo_rel_irr_bias'] in ('SFF', 'FSF', 'FFS')]; ratios = [int(r['t4']) / fast[(r['fast'], r['slow'])] for r in s if r['combo_rel_irr_bias'] in ('SFF', 'FSF', 'FFS')]
print('C-119 single replacements', singles, 'at least 2x slower:', sum(x >= 2 for x in ratios), 'of', len(ratios))
k = rows('c117_init_collapse.csv'); print('C-117 Spearman L2pr_250 vs time %.2f, L2R2S_250 %.2f' % (spearmanr([float(r['L2pr_250']) for r in k], [int(r['time']) for r in k]).statistic, spearmanr([float(r['L2R2S_250']) for r in k], [int(r['time']) for r in k]).statistic))
