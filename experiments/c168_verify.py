"""C-168 (check): recompute the numbers quoted for C-162 to C-167 from the raw outputs."""
import csv, subprocess, sys, numpy as np
def run(a): return subprocess.run([sys.executable] + a, capture_output=True, text=True).stdout.strip()
for args, saved in [(["c162_score.py"], "c162_scores.txt"), (["c164_score.py"], "c164_scores.txt")]:
    print(f"{args[0]} reproduces {saved}: {run(args) == open(saved).read().strip()}")
def med(d, arm):
    t = [int(r["t4"]) if int(r["t4"]) > 0 else 10000 for r in csv.DictReader(open(f"{d}/confine.csv")) if r["arm"] == arm]; return t, np.median(t)
for d in ["c165_d2", "c165_d4", "c166_b0.9", "c166_b0.99", "c167_rmsprop", "c167_adagrad"]:
    rows = list(csv.DictReader(open(f"{d}/confine.csv"))); arms = sorted({r["arm"] for r in rows})
    m = {a: med(d, a) for a in arms}; line = "; ".join(f"{a} {m[a][0]} (median {m[a][1]:.0f})" for a in arms)
    j = [(float(r["L2_R2S250"]), float(r["L3_R2S250"])) for r in rows if r["arm"] == "joint"]
    print(f"{d}: {line}; joint/alone {m['joint'][1] / m['alone'][1]:.2f}" + (f"; conf/alone {m['conf3'][1] / m['alone'][1]:.2f}" if "conf3" in m else "") + f"; joint R2 L2 {min(x for x, _ in j):.4f}-{max(x for x, _ in j):.4f}, L3 {min(y for _, y in j):.4f}-{max(y for _, y in j):.4f}")
