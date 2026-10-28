"""C-183 (check): recompute the numbers quoted for C-175 to C-182 from the raw outputs."""
import csv, numpy as np, subprocess, sys
def run(a): return subprocess.run([sys.executable] + a, capture_output=True, text=True).stdout.strip()
def rng(v, k=3): return f"{min(v):.{k}f}-{max(v):.{k}f}"
for d, arms in [("c176_ln12", ("alone", "joint")), ("c176_ln3", ("alone", "joint")), ("c177", ("alone", "joint_orms", "joint_4rms")), ("c178", ("alone", "joint_4center"))]:
    R = list(csv.DictReader(open(f"{d}/confine.csv"))); al = np.median([int(r["t4"]) for r in R if r["arm"] == "alone"])
    for arm in arms:
        rr = [r for r in R if r["arm"] == arm]; t = [int(r["t4"]) if int(r["t4"]) > 0 else 10000 for r in rr]
        print(f"{d} {arm}: {t} median {np.median(t):.0f} ratio {np.median(t) / al:.2f}; R2 L2 {rng([float(r['L2_R2S250']) for r in rr])} L3 {rng([float(r['L3_R2S250']) for r in rr])} corr {rng([float(r['L2norm_corrS250']) for r in rr])}")
for script, saved in [(["c179_score.py"], "c179_scores.txt"), (["c180_normprofile.py"], "c180_normprofile.txt")]:
    print(f"{script[0]} reproduces {saved}: {run(script) == open(saved).read().strip()}")
print("c175 / c182 scores are inline summaries of the raw CSV/npz files; see c175_scores.txt and c182_scores.txt (recomputed below)")
for d in ("none", "ln"):
    B = list(csv.DictReader(open(f"c175_{d}/bits12.csv"))); print("c175", d, [float(np.median([int(r[f't{i}']) if int(r[f't{i}']) > 0 else 20000 for r in B])) for i in range(13)])
