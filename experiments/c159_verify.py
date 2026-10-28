"""C-159 (check): recompute the numbers quoted for C-149 to C-158 from the raw output files. Prints each quoted value next to the recomputed one."""
import csv, numpy as np, os, subprocess, sys
def onset(st, c):  # same definition as c131_score.py
    for k in range(len(c) - 2):
        if c[k] >= 0.05 and c[k + 1] >= 0.05 and c[k + 2] >= 0.05: return int(st[k])
    return None
def rows(p): return list(csv.DictReader(open(p)))
def show(name, quoted, got): print(f"{name}: quoted {quoted} | recomputed {got}")
R = rows("c149/n12confine.csv")
for arm in ["alone", "conf3", "joint"]:
    t = [int(r["time_bit4"]) for r in R if r["arm"] == arm]; show(f"C-149 {arm}", {"alone": "400 500 400 300 400", "conf3": "400 500 500 400 400", "joint": "7,300; >20,000; 3,000; 17,900; >20,000"}[arm], t)
R = rows("c153/forcing.csv")
for br in ["block", "centered_noise", "frozen_mean", "frozen_other", "joint"]:
    show(f"C-153 {br}", "", [int(r["extra_steps_to_0.99_bit4"]) for r in R if r["branch"] == br])
for a in ["0.1", "0.3", "0.5", "0.7"]:
    T, on = [], []
    for s in range(5):
        z = np.load(f"c154/traj_a{a}_s{s}.npz"); T.append(int(z["t4"])); on.append(onset(z["steps"], z["corr"]))
    show(f"C-154 alpha {a} T / onset medians", {"0.1": "650 / 150", "0.3": "550 / 75", "0.5": "600 / 100", "0.7": "550 / 75"}[a], f"{np.median(T):.0f} / {np.median(on):.0f}")
R = rows("c156_sgd/confine.csv")
for arm in ["alone", "conf3", "joint"]:
    t = [int(r["t4"]) for r in R if r["arm"] == arm]; show(f"C-156 {arm}", {"alone": "1,800 2,100 1,700 2,400 2,250", "conf3": "1,850 2,000 1,750 2,400 2,200", "joint": "2,850 2,850 2,850 3,500 3,500"}[arm], f"{t} median {np.median(t):.0f}")
j = [float(r["L2_R2S250"]) for r in R if r["arm"] == "joint"]; show("C-156 joint L2 R2 on S at 250", "0.66-0.70", f"{min(j):.2f}-{max(j):.2f}")
t_al = np.median([int(r["t4"]) for r in R if r["arm"] == "alone"]); t_j = np.median([int(r["t4"]) for r in R if r["arm"] == "joint"]); show("C-156 joint/alone", "1.36", f"{t_j / t_al:.2f}")
for d, lr in [("c158_lr0.03", "0.03"), ("c158_lr0.003", "0.003")]:
    if os.path.exists(f"{d}/confine.csv"):
        R = rows(f"{d}/confine.csv"); al = [int(r["t4"]) for r in R if r["arm"] == "alone"]; jo = [int(r["t4"]) for r in R if r["arm"] == "joint"]
        jr = [float(r["L2_R2S250"]) for r in R if r["arm"] == "joint"]
        show(f"C-158 lr {lr} alone / joint / ratio / joint L2 R2", "see PROGRESS", f"{al} / {jo} / {np.median(jo) / np.median(al):.2f} / {min(jr):.2f}-{max(jr):.2f}")
print("\nRe-running the scoring scripts and comparing with the saved score files:")
for script, saved in [("c150_score.py", "c150_scores.txt"), ("c155_score.py", "c155_scores.txt")]:
    out = subprocess.run([sys.executable, script], capture_output=True, text=True).stdout
    same = out.strip() == open(saved).read().strip(); print(f"{script} output identical to {saved}: {same}")
for script, saved in [("c157_partA.py", "c157/partA.txt"), ("c157_partB.py", "c157/partB.txt")]:
    if os.path.exists(saved):
        out = subprocess.run([sys.executable, script], capture_output=True, text=True).stdout
        print(f"{script} output identical to {saved}: {out.strip() == open(saved).read().strip()}")
