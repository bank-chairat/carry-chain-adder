"""README Part 10: why some starting weights are lucky. Rebuilds the numbers and Figure 10 from saved results
(C-83, C-117 to C-119, C-123b/c, C-125 to C-127, C-130, C-133 to C-136, C-138, C-139, C-142 to C-147, C-152, C-161,
C-162, C-181, C-182). Run from the experiments folder: python3 summarize_part10.py"""
import glob, os, re
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from summary_utils import rows, med, where

T83 = {int(r["s"]): int(r["time"]) for r in rows("c83_a/*.csv", "c83_b/*.csv")}
v = np.array(sorted(T83.values())); print(f"C-83: 40 initial weights, same data order: joint bit-4 time {v.min()} to {v.max()} (max/min {v.max() / v.min():.1f}), median {med(v):.0f}")

def onset(st, c):   # first record with correlation >= 0.05 at three records in a row (as c131_score.py / c162_score.py)
    for k in range(len(c) - 2):
        if c[k] >= 0.05 and c[k + 1] >= 0.05 and c[k + 2] >= 0.05: return int(st[k])
    return None
def runs(pattern):
    out = {}
    for f in glob.glob(pattern):
        z = np.load(f); out[int(re.findall(r"traj_s(\d+)", f)[0])] = (z["steps"], z["corr"], int(z["t4"]))
    return out
fit, test1, test2 = runs("c125/traj_s*.npz"), runs("c127/traj_s*.npz"), runs("c162_*/traj_s*.npz")
law = lambda t_on: 1283 + 2.53 * t_on   # fitted on the even-numbered C-125 runs in C-126, then frozen
print("Onset law T = 1,283 + 2.53 x onset, tested on runs it was not fitted on:")
pts = {}
for name, R in (("C-127, 20 new initial weights", test1), ("C-162, 200 new initial weights", test2)):
    ok = [s for s in R if R[s][2] > 0 and onset(R[s][0], R[s][1]) is not None]
    th = np.array([law(onset(R[s][0], R[s][1])) for s in ok]); tt = np.array([R[s][2] for s in ok]); pts[name] = (th, tt)
    print(f"  {name}: median |ln(predicted/actual)| {np.median(np.abs(np.log(th / tt))):.2f} (a constant 2,900 gives {np.median(np.abs(np.log(2900 / tt))):.2f}), Spearman {spearmanr(th, tt).statistic:.2f}")

print("\nC-162: rank correlation between bit 4's early head start J(t) and its final time T (200 new runs)")
S = sorted(test2); T = np.array([test2[s][2] for s in S]); hz = []
for t_ in (25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500):
    J = np.array([test2[s][1][list(test2[s][0]).index(t_)] if t_ in list(test2[s][0]) else test2[s][1][-1] for s in S]); hz.append((t_, spearmanr(J, T).statistic))
print("  " + ", ".join(f"step {a}: {b:+.2f}" for a, b in hz))
auc = [(int(m.group(1)), float(m.group(2))) for m in (re.match(r"t =\s*(\d+): AUC ([\d.]+)", l) for l in open("c161_horizon.txt")) if m]
print("C-161 (from c161_horizon.txt): AUC of J(t) for 'escaped by step 500' over 1,000 initial weights: " + ", ".join(f"{a}: {b:.2f}" for a, b in auc if a in (25, 50, 100, 125, 250)))

print("\nSaved score files of the causal tests")
for f in ("c123c_scores.txt", "c130_scores.txt", "c118_scores.txt", "c119_scores.txt", "c182_scores.txt"):
    print(f"  -- {f}")
    for i, line in enumerate(open(f)):
        if f == "c182_scores.txt" and line.startswith("LN times"): continue
        if f == "c130_scores.txt" and i > 6: break
        print("    " + line.rstrip())
for f, label in (("c135/scale.csv", "position 4 x1.5, position 7 /1.5"), ("c135c/scale.csv", None), ("c135ctl/scale.csv", "control: positions 5 x1.5, 6 /1.5")):
    R = rows(f)
    for tag in dict.fromkeys(r["tag"] for r in R):
        print(f"  C-135 {tag}: times " + ", ".join(f"{r['s']}:{r['t4']}" for r in where(R, tag=tag)) + "; median ratio original/new " + f"{med([T83[int(r['s'])] / int(r['t4']) for r in where(R, tag=tag)]):.2f}")
print("\nStatic predictors on held-out initial weights (saved score files)")
for f in ("c139_scores.txt", "c142_scores.txt", "c146_scores.txt", "c147_scores.txt", "c152_scores.txt", "c181_scores.txt"):
    print(f"  -- {f}")
    for line in open(f): print("    " + line.rstrip())

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
axs[0].plot([a for a, _ in hz], [-b for _, b in hz], "o-", color="#c0392b", label="rank correlation with the final time (C-162, 200 runs; sign flipped)")
axs[0].plot([a for a, _ in auc], [2 * (b - 0.5) for a, b in auc], "s-", color="#1f4e79", label="2 x (AUC - 0.5) for escaping by step 500 (C-161, 1,000 runs)")
axs[0].set_xlabel("training step at which the head start is measured"); axs[0].set_ylim(0, 1.02)
axs[0].set_ylabel("predictive strength (0 = none, 1 = perfect)")
axs[0].set_title("(a) Luck becomes visible within ~100 steps", fontsize=10); axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
for (name, (th, tt)), col in zip(pts.items(), ("#e67e22", "#1f4e79")): axs[1].scatter(tt, th, s=12, color=col, alpha=0.7, label=name)
lim = [900, 8000]; axs[1].plot(lim, lim, "k--", lw=0.7); axs[1].set_xscale("log"); axs[1].set_yscale("log")
axs[1].set_xlabel("actual bit-4 time (steps)"); axs[1].set_ylabel("predicted from the onset (steps)")
axs[1].set_title("(b) The frozen onset law on new runs", fontsize=10); axs[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); os.makedirs("results", exist_ok=True); fig.savefig("results/part10_luck.png", dpi=150)
print("\nfigure: results/part10_luck.png")
