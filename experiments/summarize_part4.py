"""README Part 4: why are the middle bits slow? Bits trained alone versus together.
Rebuilds Table 4 and Figure 4 from saved results (C-12, C-28, C-15, C-16, C-29, C-30, C-31, C-18, C-32, C-33, C-34).
Run from the experiments folder: python3 summarize_part4.py"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summary_utils import rows, med, where, per_bit_median, fmt, t
import summarize_part1 as p1   # joint-net medians of Part 1 (prints Part 1's tables first)

print("\n================ Part 4 ================")
al8 = rows("c12/single_n8.csv"); al12 = rows("c28/single_n12.csv")
alone8 = [med([int(r["time_to_099"]) for r in al8 if r["bit"] == str(b)]) for b in range(9)]
alone12 = [med([int(r["time_to_099"]) for r in al12 if r["bit"] == str(b)]) for b in range(13)]
joint8, joint12 = p1.rows[1][1], p1.rows[4][1]
print("C-12  n = 8, each bit alone : " + fmt(alone8)); print("      n = 8, joint (C-01)    : " + fmt(joint8))
print("C-28  n = 12, each bit alone: " + fmt(alone12)); print("      n = 12, joint (C-14)   : " + fmt(joint12))

print("\nTable 4. Steps for bit 4 to reach 0.99 with different sets of co-trained bits (median of 5 seeds)")
for exp, f, cap in (("C-15", "c15/subsets_n8.csv", 10000), ("C-16", "c16/subsets_n8.csv", 10000), ("C-29", "c29/subsets_n8.csv", 10000),
                    ("C-30", "c30/subsets_n8.csv", 10000), ("C-31 (n = 12)", "c31/subsets_n12.csv", 20000)):
    R = rows(f); sets = list(dict.fromkeys(r["set"] for r in R))
    print(f"  {exp}: " + " | ".join(f"{{{s.replace('-', ',')}}} {med([t(r['time_bit4'], cap) for r in R if r['set'] == s]):,.0f}" for s in sets))

print("\nC-18: loss weight w on bit 4 (bit-4 median) ; C-32: weight 0.1 on bits 5-8")
w18 = rows("c18/weights_n8.csv")
print("  " + " | ".join(f"w = {w}: {med([t(r['t4'], 10000) for r in where(w18, w=w)]):,.0f}" for w in ("1", "3", "10", "30")))
w32 = rows("c32/weights_n8.csv")
for w in ("0", "1"):
    print(f"  C-32 {'uniform' if w == '0' else 'bits 5-8 x0.1'}: per-bit medians " + fmt(per_bit_median(where(w32, w=w), 10000)))

print("\nC-33 / C-34: k shared hidden layers (rest private per bit), and one wide shared net")
tr = rows("c33/trunk_n8.csv"); wd = rows("c34/wide_n8.csv")
for k in ("0", "1", "2", "3"):
    R = where(tr, k=k); print(f"  k = {k}: bit 4 {med([t(r['t4'], 10000) for r in R]):,.0f}, slowest bit {med([max(t(r[f't{i}'], 10000) for i in range(9)) for r in R]):,.0f}")
print(f"  width-384 fully shared: bit 4 {med([t(r['t4'], 10000) for r in wd]):,.0f}, slowest bit {med([max(t(r[f't{i}'], 10000) for i in range(9)) for r in wd]):,.0f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, (j, a, n, src) in zip(axs, ((joint8, alone8, 8, "C-01 / C-12"), (joint12, alone12, 12, "C-14 / C-28"))):
    ax.plot(range(n + 1), j, "o-", color="#c0392b", label="all bits trained together")
    ax.plot(range(n + 1), a, "s-", color="#1f4e79", label="each bit trained alone")
    ax.set_yscale("log"); ax.set_title(f"n = {n} ({src})"); ax.set_xlabel("sum bit")
axs[0].set_ylabel("steps until learned (median, log scale)"); axs[0].legend(frameon=False, fontsize=9)
fig.suptitle("The hump appears only when the bits share one network"); fig.tight_layout()
os.makedirs("results", exist_ok=True); fig.savefig("results/part4_joint_vs_alone.png", dpi=150)
print("\nfigure: results/part4_joint_vs_alone.png")
