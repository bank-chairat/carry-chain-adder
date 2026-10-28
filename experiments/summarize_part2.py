"""README Part 2: does the difficulty of a bit follow the carry chain?
Rebuilds the numbers of Part 2 from saved results (C-06, C-08, C-100, C-157, C-160) and draws Figure 2.
The margin checks (R-1, V-1, R-5, V-4, R-8, R-9) are in part2_margin_checks.py.
Run from the experiments folder: python3 summarize_part2.py"""
import csv, glob, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# C-06 (2): while a bit is half learned, is it wrong more often when a carry reaches it?
acc, bins = {}, defaultdict(lambda: [0.0, 0])
for r in csv.DictReader(open("c06/curves.csv")):
    if r["op"] == "add": acc[(r["n"], r["frac"], r["seed"], r["step"], r["bit"])] = float(r["test_acc"])
for r in csv.DictReader(open("c06/bins.csv")):
    if r["op"] != "add": continue
    k = (r["n"], r["frac"], r["seed"], r["step"], r["bit"]); c = bins[k + ("carry" if r["bin"] != "0" else "none",)]
    c[0] += float(r["test_error"]) * int(r["n_examples"]); c[1] += int(r["n_examples"])
mid = [k for k, v in acc.items() if 0.60 <= v <= 0.95]   # cases with no example in one group count as 'not higher'
rate = lambda c: c[0] / max(c[1], 1)
higher = sum(rate(bins[k + ("carry",)]) > rate(bins[k + ("none",)]) for k in mid)
print(f"C-06: in {len(mid)} half-learned (seed, bit, step) cases, error is higher with a carry in {100 * higher / len(mid):.1f}%")

# C-08: same number of training pairs, random against stratified by the longest run
print("\nC-08: held-out error on bits 5-8 with dependency length >= 4, and mean accuracy (seeds 30-34)")
acc8 = {(r["seed"], r["scheme"]): float(r["test_acc_all_bits_mean"]) for r in csv.DictReader(open("c08/summary.csv"))}
for scheme in ("random", "stratified"):
    errs = []
    for s in range(30, 35):
        z = np.load(f"c08/final_{s}_{scheme}.npz"); e, d = z["err"], z["dep"]
        m = np.stack([d[:, i] >= 4 for i in range(5, 9)], 1); errs.append(e[:, 5:9][m].mean())
    print(f"  {scheme:10s} error " + " ".join(f"{x:.4f}" for x in errs) + "  | accuracy " + " ".join(f"{acc8[(str(s), scheme)]:.4f}" for s in range(30, 35)))

# C-100: errors left at the moment a bit first passes 0.99 (saved score file of c100_score.py)
print("\nC-100 (from c100_score.py): ratio of error on the longest active carry chain to error with no carry, at the first 0.99")
for line in open("c100_scores.txt"):
    if line.startswith("bit 4 seed") or "median ratio" in line: print("  " + line.strip())

# C-157 / C-160: equal training counts per bit's own dependency length
def err_by_d(pattern, bit):
    e = defaultdict(int); n = defaultdict(int)
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        for d in range(bit + 1): sel = z["dep"][:, bit] == d; e[d] += z["err"][sel, bit].sum(); n[d] += sel.sum()
    return [e[d] / n[d] for d in range(bit + 1)]
curves = {}
for bit, own in ((6, "c157/final_*_own6.npz"), (5, "c160/final_*_own5.npz")):
    print(f"\nBit-{bit} held-out error by its own dependency length d = 0..{bit}, pooled over seeds 30-34")
    for name, pat in (("random", "c08/final_*_random.npz"), ("stratified by longest run", "c08/final_*_stratified.npz"), (f"equal count per d (own{bit})", own)):
        curves[(bit, name)] = err_by_d(pat, bit); print(f"  {name:28s} " + " ".join(f"{x:.4f}" for x in curves[(bit, name)]))
print("  (logistic-regression slopes with bootstrap intervals: c157_partB.py, BIT=5 c157_partB.py)")

fig, ax = plt.subplots(figsize=(6.6, 4))
for name, col, mk in (("random", "#c0392b", "o"), ("stratified by longest run", "#e67e22", "s"), ("equal count per d (own6)", "#1f4e79", "^")):
    y = np.array(curves[(6, name)]) * 100
    ax.plot(range(7), y, marker=mk, color=col, label={"random": "random training pairs",
            "stratified by longest run": "balanced by the pair's longest run (C-08)",
            "equal count per d (own6)": "equal count for every length of bit 6 (C-157)"}[name])
ax.set_xlabel("dependency length of bit 6 (how far a carry must travel to reach it)")
ax.set_ylabel("held-out error of sum bit 6 (%)"); ax.set_title("Errors follow how often a case is trained, not its length")
ax.legend(frameon=False, fontsize=9); fig.tight_layout()
os.makedirs("results", exist_ok=True); fig.savefig("results/part2_error_vs_chain.png", dpi=150)
print("\nfigure: results/part2_error_vs_chain.png")
