"""README Part 3: is a carry represented inside the network, and what circuit is it?
Rebuilds Tables 2 and 3 and Figure 3 from saved results (C-04, C-81, C-90/C-93, C-91/C-93, C-94, C-95, C-151).
Run from the experiments folder: python3 summarize_part3.py"""
import os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summary_utils import rows, med

print("C-04: smallest hidden width (on the grid) at which at least 3 of 5 seeds fit every pair exactly")
r = rows("c04a/runs.csv", "c04b/runs.csv"); ok = defaultdict(int)
for x in r: ok[(x["op"], int(x["n"]), int(x["depth"]), int(x["width"]))] += int(x["success"])
for op in ("add", "xor"):
    for depth in (1, 2):
        ws = [min([w for (o, n_, d, w), k in ok.items() if o == op and n_ == n and d == depth and k >= 3], default=None) for n in (3, 4, 5, 6)]
        print(f"  {op} depth {depth}: n = 3..6 -> {ws}")

print("\nC-81: exact Walsh spectrum of each sum bit (from c81_walsh.txt, written by c81_walsh.py)")
for line in open("c81_walsh.txt"):
    f = line.split()
    if len(f) > 8 and f[1] == "W0-2": print(f"  bit {f[0]}: weight at degrees 0-2 {f[2]}, mean degree {f[4]}, non-zero coefficients {f[8]}")

print("\nC-90 / C-93: accuracy drop per sum bit 0..8 after mean-ablating layer-1 unit groups (mean over 10 nets)")
ab = rows("c90/ablate_n8.csv", "c93_ablate/ablate_n8.csv"); drops = {}
for g in ("G", "T", "R"):
    drops[g] = [float(np.mean([float(x[f"drop{i}"]) for x in ab if x["group"] == g])) for i in range(9)]
    print(f"  {g}: " + " ".join(f"{v:.3f}" for v in drops[g]))
print("  units in G per net: " + " ".join(x["nG"] for x in ab if x["group"] == "G"))

print("\nC-91 / C-93: patching flip rate (activations of the group taken from an input with the other carry)")
pa = rows("c91/patch_n8.csv", "c93_patch/patch_n8.csv")
for nets, label in ((range(0, 5), "first 5 nets (C-91)"), (range(5, 10), "fresh 5 nets (C-93)")):
    for bit in (3, 4, 5):
        m = {s: float(np.mean([float(x["flip"]) for x in pa if x["set"] == s and x["bit"] == str(bit) and int(x["seed"]) in nets])) for s in ("G", "T", "R")}
        print(f"  {label}, bit {bit}: G {m['G']:.2f}  T {m['T']:.2f}  R {m['R']:.2f}")

print("\nC-94: flip rate when the top 8 units of one layer are patched (ten nets)")
ly = rows("c94/layers_n8.csv")
for bit in ("4", "5"):
    print(f"  bit {bit}: " + "  ".join(f"layer {L} top {np.mean([float(x['flip']) for x in ly if x['bit'] == bit and x['layer'] == L and x['m'] == '8' and x['set'] == 'top']):.2f} / random {np.mean([float(x['flip']) for x in ly if x['bit'] == bit and x['layer'] == L and x['m'] == '8' and x['set'] == 'rand']):.2f}" for L in ("1", "2", "3")))

print("\nC-95 (n = 12): ablation drop averaged over bits 4-9, and flip rate at bit 7")
a12 = rows("c95_*/ablate_n12.csv"); p12 = rows("c95_*/patch_n12.csv")
for g in ("G", "T", "R"):
    print(f"  {g}: drop {np.mean([np.mean([float(x[f'drop{i}']) for i in range(4, 10)]) for x in a12 if x['group'] == g]):.3f}  flip {np.mean([float(x['flip']) for x in p12 if x['set'] == g and x['bit'] == '7']):.2f}")

print("\nC-151 (from c151b_scores.txt): what layers 2 and 3 hold after training")
for line in open("c151b_scores.txt"): print("  " + line.rstrip())

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(9); w = 0.27
for k, (g, col, lab) in enumerate((("G", "#1f4e79", "about 11 units with doubling weights (G)"), ("T", "#e67e22", "same number of largest other units (T)"), ("R", "#95a5a6", "same number of random units (R)"))):
    ax.bar(x + (k - 1) * w, drops[g], w, color=col, label=lab)
ax.set_xticks(x); ax.set_xlabel("sum bit (8 = carry out)"); ax.set_ylabel("accuracy lost when the units are removed")
ax.set_title("A small group of first-layer units is needed for the middle bits"); ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); os.makedirs("results", exist_ok=True); fig.savefig("results/part3_unit_ablation.png", dpi=150)
print("\nfigure: results/part3_unit_ablation.png")
