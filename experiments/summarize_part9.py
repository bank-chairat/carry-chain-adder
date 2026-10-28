"""README Part 9: width, initial scale and learning rate, and the search for a formula for the delay.
Rebuilds Table 11 and Figure 9 from saved results (C-63, C-66 to C-80, C-86, C-98, C-103, C-122, C-131, C-140, C-141,
C-150, C-154, C-179). Run from the experiments folder: python3 summarize_part9.py"""
import os, re
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summary_utils import rows, med, where, t, fmt

def j4(f, cap=12000, lam="1.0"): return med([t(r["t4"], cap) for r in rows(f) if r.get("lam", lam) == lam])

print("Table 11. Joint bit-4 time (median of 5 seeds; batch 4,096 for the scale rows)")
print("  C-66 initial scale c = 0.5 / 1 / 2 / 4: joint " + fmt([j4(f"c66_c{d}/initscale_c*_n8.csv") if c != "1.0" else j4("c65_bs4096/gradshare_bs4096_n8.csv") for c, d in (("0.5", "0.5"), ("1.0", "1"), ("2.0", "2"), ("4.0", "4"))])
      + "; alone " + fmt([j4(f"c66_c{d}/initscale_c*_n8.csv", 1500, "0.0") if c != "1.0" else j4("c65_bs4096/gradshare_bs4096_n8.csv", 1500, "0.0") for c, d in (("0.5", "0.5"), ("1.0", "1"), ("2.0", "2"), ("4.0", "4"))]))
for c in ("0.5", "1", "2", "4"):
    R = rows(f"c67_c{c}/lazy_c*_n8.csv"); print(f"  C-67 c = {c}: relative weight change at the bit-4 time, layer 1 / 2 / 3 / head: " + " / ".join(f"{med([float(r[k]) for r in R]):.2f}" for k in ("rel1", "rel2", "rel3", "relhead")))
print("  C-69 (hidden x2, output x1) / (hidden x1, output x8) / (hidden x2, output x1/8): " + fmt([j4(f"c69_{k}/*.csv") for k in "ABC"]))
print("  C-70 hidden-layer learning rate x1 / x2 / x4 / x8: " + fmt([j4("c65_bs4096/gradshare_bs4096_n8.csv")] + [j4(f"c70_k{k}/*.csv") for k in ("2", "4", "8")]))
import csv, json
for c in ("2", "4"):
    R = [[20000 if x < 0 else x for x in json.loads(r["bit_times_json"])] for r in rows(f"c72_c{c}/*.csv")]
    print(f"  C-72 n = 12, scale {c}: slowest bit per seed median {med([max(r) for r in R]):,.0f}, censored cells {sum(x == 20000 for r in R for x in r)}")

print("\nC-63: Adam dilution factor and the part it leaves unexplained, by width")
for W, f in (("64", "c63_w64/dilution_w64_n8.csv"), ("128", "c61/dilution_n8.csv"), ("256", "c63_w256/dilution_w256_n8.csv"), ("512", "c63_w512/dilution_w512_n8.csv")):
    R = rows(f); print(f"  W = {W}: median s over the first 500 steps {med([float(r['s_now']) for r in R if r['lam'] == '1.0' and int(r['step']) <= 500 and r['t4_flag'] == '0']):.3f}")
print("C-74 separate moments, bit-4 median at W = 64 / 128 / 512: " + fmt([j4("c74_w64/sepmoments_n8.csv", 6000), j4("c73_f1/sepmoments_n8.csv", 6000), j4("c74_w512/sepmoments_n8.csv", 6000)]))

pb = rows("c75/init_probe.csv")
print("\nC-75: a4 XOR b4 readability of untrained nets (best layer, median): " + ", ".join(f"W {W}: {med([max(float(r['probe_acc']) for r in pb if r['W'] == W and r['seed'] == s and r['target'] == 'p4') for s in '01234']):.3f}" for W in ("64", "128", "256", "512", "1024")))
print("C-76 mixed widths: " + ", ".join(f"{k}: {j4(f'c76_{k}/mixed_n8.csv'):,.0f}" for k in ("512_64_64", "64_512_512", "64_64_512")))
print("C-77/C-78 frozen random layer in front (width 128 / 512 / 1024): " + fmt([j4("c78_fw128/frozen_n8.csv"), j4("c78_fw512/frozen_n8.csv"), j4("c77_joint/frozen_n8.csv")]) + f"; alone {j4('c77_alone/frozen_n8.csv', 12000, '0.0'):,.0f}")
print("C-79 plain depth 2 / 4 x 128: " + fmt([j4("c79_2x128/mixed_n8.csv"), j4("c79_4x128/mixed_n8.csv")]))
for W in ("64", "128", "512"):
    R = rows(f"c80_w{W}/lazy_w{W}_n8.csv"); print(f"C-80 W = {W}: layer-1 relative change at the bit-4 time, joint {med([float(r['rel1']) for r in where(R, lam='1.0')]):.2f}, alone {med([float(r['rel1']) for r in where(R, lam='0.0')]):.2f}")

print("\nC-86 / C-98 / C-103: the gradient-L1 formula and why it is only a description")
for f in ("c86_formula.txt", "c98_scores.txt", "c103_scale_comp.txt"):
    for line in open(f): print("  " + line.rstrip())

print("\nC-122 / C-131 / C-154: the two halves of the width effect")
w = rows("c122/widthdup.csv")
for a in ("W128dup4", "DUP512"): print(f"  {a}: alone {med([t(r['t4'], 10000) for r in w if r['arm'] == a and r['mode'] == 'alone']):,.0f}, joint {med([t(r['t4'], 10000) for r in w if r['arm'] == a and r['mode'] == 'joint']):,.0f}")
traj = {}
for line in open("c131_scores.txt"):
    m = re.match(r"(\w+)\s+T \[.*\] t_on \[.*\] D \[.*\] \| medians T (\d+) t_on (\d+) D (\d+)", line)
    if m: traj[m.group(1)] = tuple(int(x) for x in m.groups()[1:]); print("  " + line.rstrip())
mix = {}
for line in open("c154_scores.txt"):
    m = re.match(r"alpha ([\d.]+): .*medians T (\d+) onset (\d+) growth (\d+)", line)
    if m: mix[float(m.group(1))] = tuple(int(x) for x in m.groups()[1:]); print("  " + line.rstrip())

print("\nC-140 / C-141 / C-150 / C-179 (saved score files)")
for f in ("c140_scores.txt", "c141_scores.txt", "c150_scores.txt", "c179_scores.txt"):
    print(f"  -- {f}")
    for line in open(f): print("    " + line.rstrip())

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
names = ["width 128", "width 128 copied 4 times\n(same units, Adam steps x4)", "width 512, independent"]
vals = [traj["W128"], traj["DUP512"], traj["W512"]]
x = np.arange(3); axs[0].bar(x - 0.2, [v[1] for v in vals], 0.4, color="#e67e22", label="onset (bit 4's signal appears)")
axs[0].bar(x + 0.2, [v[2] for v in vals], 0.4, color="#1f4e79", label="growth (from onset to learned)")
axs[0].set_xticks(x, names, fontsize=8.5); axs[0].set_ylabel("steps (median of 5 seeds)"); axs[0].legend(frameon=False, fontsize=8.5)
axs[0].set_title("(a) Two halves of the width effect (C-131)", fontsize=10)
al = sorted(mix); axs[1].plot(al, [mix[a][1] for a in al], "o-", color="#e67e22", label="onset"); axs[1].plot(al, [mix[a][2] for a in al], "s-", color="#1f4e79", label="growth")
axs[1].set_xlabel("share of independent weights mixed into the copied net (amplitude)"); axs[1].set_ylabel("steps (median)")
axs[1].set_title("(b) A little independence brings the onset forward (C-154)", fontsize=10); axs[1].legend(frameon=False, fontsize=8.5)
fig.tight_layout(); os.makedirs("results", exist_ok=True); fig.savefig("results/part9_width.png", dpi=150)
print("\nfigure: results/part9_width.png")
