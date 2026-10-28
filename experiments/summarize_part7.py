"""README Part 7: the takeover. The fast upper bits turn the shared layers into a code of the sum, and a
per-parameter normalised optimiser turns that into a delay for bit 4. Rebuilds Table 8 and Figure 7 from saved
results (C-109 to C-116, C-121, C-129, C-129b, C-132, C-149, C-153, C-156, C-158, C-165 to C-167).
Run from the experiments folder: python3 summarize_part7.py"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summary_utils import rows, med, where, t

def extra(f, key="branch"):
    R = rows(f); v = lambda x: 10 ** 9 if int(x) < 0 else int(x)   # -1 = not reached within the cap
    out = {b: med([v(r["extra_steps_to_0.99_bit4"]) for r in R if r[key] == b]) for b in dict.fromkeys(r[key] for r in R)}
    return {b: ("not reached" if x >= 10 ** 9 else int(x)) for b, x in out.items()}
h = rows("c109/history.csv")
print("C-109: extra recovery steps after K joint steps: " + ", ".join(f"K = {K}: {med([int(r['extra_steps_to_0.99_bit4']) for r in h if r['branch'].startswith(f'K{K}_')]):.0f}" for K in (100, 250, 500, 1000)))
print("C-110: after 250 steps of pre-training on: " + str(extra("c110/kinds.csv", "pretrain")))

d = rows("c111_a/diag.csv", "c111_b/diag.csv")
print("\nC-111: medians by state (layer-2 participation ratio, layer-3 effective rank, layer-2 norm ratio, cost)")
for c in dict.fromkeys(r["cond"] for r in d):
    R = where(d, cond=c); print(f"  {c:13s} L2 PR {med([float(r['L2_pr']) for r in R]):5.1f}  L3 erank {med([float(r['L3_erank']) for r in R]):5.1f}  L2 norm {med([float(r['L2_normratio']) for r in R]):.2f}  cost {med([int(r['cost']) for r in R]):.0f}")

g = rows("c112/norm_geom.csv"); none = {0: 250, 1: 200, 2: 300, 3: 250, 4: 300}   # C-111 'none' costs
D = {(r["arm"], int(r["seed"])): int(r["cost"]) for r in g}
print("\nC-112: share of the extra cost removed: " + ", ".join(f"{a} {med([(D[('base', s)] - D[(a, s)]) / (D[('base', s)] - none[s]) for s in range(5)]):.2f}" for a in ("N_all", "N_L2", "D_L2", "R_L2")))

m = rows("c113a.csv")
print("C-113a: layer-2 / layer-3 R^2 on the sum S after 250 steps: " + ", ".join(f"{k} {med([float(r['L2_R2S']) for r in where(m, kind=k)]):.2f}/{med([float(r['L3_R2S']) for r in where(m, kind=k)]):.2f}" for k in dict.fromkeys(r["kind"] for r in m)))
c = rows("c113b_a/companions.csv", "c113b_b/companions.csv")
print("C-113b: bit-4 time with four companion outputs: " + ", ".join(f"{k} {med([int(r['t4']) for r in where(c, cond=k)]):.0f}" for k in ("alone", "upper4", "Sthr4", "Rthr4")))

b = rows("c114b_a/sepblock.csv", "c114b_b/sepblock.csv")
print("\nC-114b (separate moments): bit-4 time when the other outputs may not update the listed layers")
for k in ("none", "L12", "L1", "L2", "L3", "all"):
    R = where(b, blocked_layers=k); print(f"  {k:4s}: {med([t(r['t4'], 6000) for r in R]):6,.0f}   layer-2 R^2 on S at 250: {med([float(r['L2_R2S250']) for r in R]):.2f}")
from scipy.stats import spearmanr
print(f"  Spearman(layer-2 R^2 on S, time) over all runs {spearmanr([float(r['L2_R2S250']) for r in b], [t(r['t4'], 6000) for r in b]).statistic:+.2f}")

r115 = rows("c115/rotation.csv"); C = {(r["start"], int(r["seed"]), r["m"]): int(r["cost"]) for r in r115}
ex = lambda s, mm: C[("base", s, mm)] - C[("D_L2", s, mm)]
print("C-115: extra(m)/extra(1) for layer-2 step multiplier m (seeds 0,1,3,4): " + ", ".join(f"m = {mm}: {med([ex(s, mm) / ex(s, '1.0') for s in (0, 1, 3, 4)]):.2f}" for mm in ("0.5", "2.0", "4.0")))

print("\nTable 8. Ordinary Adam: the other outputs only in the last hidden layer (conf3) against joint (C-116, C-149, C-165)")
settings = [("width 64", "c116_W64_FW0", 10000), ("width 128", "c116_W128_FW0", 10000), ("width 512", "c116_W512_FW0", 10000),
            ("frozen random layer", "c116_W128_FW1024", 10000), ("initial scale 2", "c116_W128_CS2", 10000), ("initial scale 4", "c116_W128_CS4", 10000),
            ("lr 3e-4", "c116_W128_lr3e-4", 10000), ("lr 1e-2", "c116_W128_lr1e-2", 3000), ("depth 2", "c165_d2", 10000), ("depth 4", "c165_d4", 10000)]
ratios = []
for name, f, cap in settings:
    R = rows(f"{f}/confine.csv"); mm = {a: med([t(r["t4"], cap) for r in where(R, arm=a)]) for a in ("alone", "conf3", "joint")}
    ratios.append((name, mm["conf3"] / mm["alone"], mm["joint"] / mm["alone"]))
    print(f"  {name:20s} alone {mm['alone']:5,.0f}  conf3 {mm['conf3']:5,.0f}  joint {mm['joint']:6,.0f}   conf3/alone {ratios[-1][1]:.2f}  joint/alone {ratios[-1][2]:.1f}")
R = rows("c149/n12confine.csv")
print("  n = 12 (C-149)       " + "  ".join(f"{a} {med([t(r['time_bit4'], 20000) for r in where(R, arm=a)]):,.0f}" for a in ("alone", "conf3", "joint")))

print("\nC-121 / C-129 / C-129b / C-132")
p = rows("c121_plain/layer3.csv"); print("  ordinary Adam: " + ", ".join(f"{a} {med([t(r['t4'], 6000) for r in where(p, arm=a)]):,.0f}" for a in ("joint", "blockL3")))
l3 = rows("c129/l3dil.csv"); print("  C-129: " + ", ".join(f"{a} {med([t(r['t4'], 6000) for r in where(l3, arm=a)]):,.0f}" for a in dict.fromkeys(r["arm"] for r in l3)))
sb = rows("c129b/sepblock.csv"); print("  C-129b others' steps x1/8: " + ", ".join(f"{a} {med([t(r['t4'], 6000) for r in where(sb, blocked_layers=a)]):,.0f}" for a in ("none", "L3")))
print("  C-132 step ratio r = 1, 2, 4, 8, 16: " + ", ".join(f"{med([t(r['t4'], 6000) for r in where(rows(*f), blocked_layers='none')]):,.0f}" for f in (("c129b/sepblock.csv",), ("c132_FO0.25/sepblock.csv",), ("c132_FO0.5/sepblock.csv",), ("c114_a/sepblock.csv", "c114_b/sepblock.csv"), ("c132_FO2/sepblock.csv",))))

print("C-153: extra steps after 500 joint steps with different forces on the body: " + str(extra("c153/forcing.csv")))

print("\nOptimisers: joint/alone bit-4 ratio")
opt = [("Adam lr 1e-3", "c116_W128_FW0", 10000)] + [("RMSprop lr 1e-3", "c167_rmsprop", 10000), ("Adagrad lr 1e-2", "c167_adagrad", 10000),
       ("Adam beta2 0.99", "c166_b0.99", 10000), ("Adam beta2 0.9", "c166_b0.9", 10000),
       ("SGD lr 0.003", "c158_lr0.003", 20000), ("SGD lr 0.01", "c156_sgd", 10000), ("SGD lr 0.03", "c158_lr0.03", 10000)]
orat = []
for name, f, cap in opt:
    R = rows(f"{f}/confine.csv"); a, j = med([t(r["t4"], cap) for r in where(R, arm="alone")]), med([t(r["t4"], cap) for r in where(R, arm="joint")])
    orat.append((name, j / a)); print(f"  {name:16s} alone {a:5,.0f} joint {j:6,.0f} ratio {j / a:5.1f}   joint layer-2 R^2 on S at 250: {min(float(r['L2_R2S250']) for r in where(R, arm='joint')):.2f}-{max(float(r['L2_R2S250']) for r in where(R, arm='joint')):.2f}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
y = np.arange(len(ratios))[::-1]
axs[0].barh(y + 0.2, [r[2] for r in ratios], 0.4, color="#c0392b", label="all outputs share every layer")
axs[0].barh(y - 0.2, [r[1] for r in ratios], 0.4, color="#1f4e79", label="other outputs only in the last hidden layer")
axs[0].set_yticks(y, [r[0] for r in ratios], fontsize=9); axs[0].set_xscale("log"); axs[0].axvline(1, color="k", lw=0.6)
axs[0].set_xlabel("bit-4 time / bit-4 time alone (log; censored runs at the cap)"); axs[0].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1)
axs[0].set_title("(a) Keeping the others out of the lower layers removes the delay", fontsize=10)
y2 = np.arange(len(orat))[::-1]
axs[1].barh(y2, [r[1] for r in orat], color=["#c0392b"] * 5 + ["#1f4e79"] * 3); axs[1].set_yticks(y2, [r[0] for r in orat], fontsize=9)
axs[1].set_xscale("log"); axs[1].axvline(1, color="k", lw=0.6); axs[1].set_xlabel("joint / alone (log)")
axs[1].set_title("(b) Large only with per-parameter normalised optimisers", fontsize=10); fig.tight_layout()
os.makedirs("results", exist_ok=True); fig.savefig("results/part7_takeover.png", dpi=150)
print("\nfigure: results/part7_takeover.png")
