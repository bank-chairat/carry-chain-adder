"""README Part 6: where does bit 4's slowdown come from? The other outputs' gradient and Adam.
Rebuilds the numbers and Figure 6 from saved results (C-58 to C-65, C-73, C-97, C-101, C-104, C-107, C-108).
Run from the experiments folder: python3 summarize_part6.py"""
import csv, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summary_utils import rows, med, where, t, fmt

# C-58: when does a4 XOR b4 become linearly readable (best layer >= 0.95 at two checkpoints in a row)?
pr = defaultdict(dict)
for r in csv.DictReader(open("c58/probe_time.csv")):
    if r["target"] == "p4":
        k = (r["net"], r["seed"], int(r["step"])); pr[k[:2]][k[2]] = max(pr[k[:2]].get(k[2], 0), float(r["probe_acc"]))
print("C-58: step at which a4 XOR b4 becomes readable (median over 5 seeds; never = inf)")
for net in ("single", "lower", "upper", "joint"):
    ts = []
    for s in range(5):
        c = sorted(pr[(net, str(s))].items()); ts.append(next((a for (a, x), (_, y) in zip(c, c[1:]) if x >= 0.95 and y >= 0.95), np.inf))
    print(f"  {net:6s}: {[int(x) if np.isfinite(x) else None for x in ts]}, median {med(ts)}")

print("\nC-59 / C-60: bit-4 time when the other outputs' gradient into the shared layers is scaled by lam")
lam_t = {}
for opt, f in (("Adam", "c59/gradshare_n8.csv"), ("momentum SGD", "c60/gradshare_sgd_n8.csv")):
    R = rows(f); lam_t[opt] = [med([t(r["t4"], 10000) for r in where(R, lam=l)]) for l in ("0.0", "0.03", "0.1", "0.3", "1.0")]
    print(f"  {opt:12s} lam 0 / 0.03 / 0.1 / 0.3 / 1: " + fmt(lam_t[opt]) + f"  (joint/alone {lam_t[opt][-1] / lam_t[opt][0]:.2f})")

d = rows("c61/dilution_n8.csv")
for lam in ("1.0", "0.3"):
    print(f"C-61: Adam dilution factor s over the first 500 steps, lam {lam}: median {med([float(r['s_now']) for r in d if r['lam'] == lam and int(r['step']) <= 500 and r['t4_flag'] == '0']):.3f}")

v = rows("c62/variance_n8.csv"); T = {(r["init"], r["order"]): t(r["t4"], 8000) for r in v}
y = np.log([T[(i, o)] for i in "01234" for o in "012"]).reshape(5, 3)
tot = ((y - y.mean()) ** 2).sum()   # two-way layout: 5 initial weights x 3 batch orders
print(f"\nC-62: share of the variance of log(bit-4 time) between initial weights {3 * ((y.mean(1) - y.mean()) ** 2).sum() / tot:.2f}, between batch orders {5 * ((y.mean(0) - y.mean()) ** 2).sum() / tot:.2f}")

print("\nC-65: batch size (joint / alone bit-4 median)")
for bs, f in (("64", "c65_bs64/gradshare_bs64_n8.csv"), ("256", "c59/gradshare_n8.csv"), ("1024", "c65_bs1024/gradshare_bs1024_n8.csv"), ("4096", "c65_bs4096/gradshare_bs4096_n8.csv")):
    R = rows(f); al = [t(r["t4"], 1500) for r in where(R, lam="0.0")]
    print(f"  batch {bs}: joint {med([t(r['t4'], 12000) for r in where(R, lam='1.0')]):,.0f}" + (f", alone {med(al):,.0f}" if al else ""))

print("\nC-73: separate Adam moments per output in the shared layers (f = size of each output's step)")
for f in ("1", "0.333"):
    try: R = rows(f"c73_f{f}/sepmoments_n8.csv")
    except FileNotFoundError: R = rows(f"c73_f{'0.3333333333333333' if f == '0.333' else f}/sepmoments_n8.csv")
    print(f"  f = {f}: bit 4 " + fmt([t(r['t4'], 6000) for r in R]) + f", median {med([t(r['t4'], 6000) for r in R]):,.0f}")

print("\nC-97 (from c97_scores.txt): joint/alone ratio against the learning rate")
lr_ratio = []
for line in open("c97_scores.txt"):
    f = line.split(); lr_ratio.append((float(f[0]), float(line.split("median ratio ")[1].split()[0]))); print("  " + f"lr {f[0]}: median ratio {lr_ratio[-1][1]}")

print("\nC-101: curvature along bit 4's most negative direction (Adam metric), 9 checkpoints")
for r in rows("c101_hessian.csv"):
    print(f"  seed {r['seed']} step {r['step']}: bit 4 {float(r['qS4q']):7.1f}, others {float(r['qSoq']):6.1f}, joint {float(r['joint']):6.1f}, Gauss-Newton share {float(r['gn_frac']):.2f}, gradient cosine {float(r['cos']):+.3f}")

print("\nC-104 / C-107 / C-108: after 500 joint steps, extra steps for bit 4 to recover with only its own body gradient")
for f, keys in (("c104/history.csv", None), ("c107/adamparts.csv", None), ("c108/adamparts.csv", None)):
    R = rows(f); br = list(dict.fromkeys(r["branch"] for r in R))
    print(f"  {f.split('/')[0]}: " + " | ".join(f"{b} {med([int(r['extra_steps_to_0.99_bit4']) for r in where(R, branch=b)]):.0f}" for b in br))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
lams = [0, 0.03, 0.1, 0.3, 1]
for opt, col in (("Adam", "#c0392b"), ("momentum SGD", "#1f4e79")): axs[0].plot(lams, lam_t[opt], "o-", color=col, label=opt)
axs[0].set_xscale("symlog", linthresh=0.03); axs[0].set_xticks(lams, ["0", "0.03", "0.1", "0.3", "1"])
axs[0].set_xlabel("share of the other outputs' gradient reaching the shared layers"); axs[0].set_ylabel("steps until bit 4 is learned (median)")
axs[0].set_title("(a) The other outputs' gradient slows bit 4 under Adam"); axs[0].legend(frameon=False)
x, yv = zip(*lr_ratio); axs[1].plot(x, yv, "s-", color="#c0392b"); axs[1].set_xscale("log")
axs[1].set_xlabel("Adam learning rate"); axs[1].set_ylabel("joint time / time alone (median)")
axs[1].set_title("(b) Not a simple rescaling of time"); fig.tight_layout()
os.makedirs("results", exist_ok=True); fig.savefig("results/part6_gradient_share.png", dpi=150)
print("\nfigure: results/part6_gradient_share.png")
