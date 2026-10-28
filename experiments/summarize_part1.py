"""README Part 1: in what order are the sum bits learned?
Rebuilds Table 1 and Figure 1 from the saved learning curves (C-01, C-06, R-2, C-14, C-19).
Run from the experiments folder: python3 summarize_part1.py"""
import csv, json, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def bit_times(path, keys, thr=0.99, cap=None):
    """Learning time of each bit: first evaluated step with held-out accuracy >= thr that is still >= thr
    at the next evaluation. Returns {key tuple: {seed: [time per bit]}}; never reached -> cap (or None)."""
    curves = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        curves[tuple(r[k] for k in keys) + (r["seed"], int(r["bit"]))][0].append((int(r["step"]), float(r["test_acc"])))
    out = defaultdict(dict)
    for (*key, seed, bit), d in curves.items():
        c = sorted(d[0]); t = cap
        for (s, a), (_, a2) in zip(c, c[1:]):
            if a >= thr and a2 >= thr: t = s; break
        out[tuple(key)].setdefault(seed, {})[bit] = t
    return {k: {s: [v[b] for b in sorted(v)] for s, v in d.items()} for k, d in out.items()}

def med(rows): return [float(np.median(col)) for col in zip(*rows)]

c01 = bit_times("c01/curves.csv", ["n", "op"], cap=10000)
c06 = bit_times("c06/curves.csv", ["n", "frac", "op"], cap=10000)
def json_runs(path, cap):
    return [[cap if t < 0 else t for t in json.loads(r["bit_times_json"])] for r in csv.DictReader(open(path))]
n12 = json_runs("c14/runs_add_adam_n12.csv", 20000); n16 = json_runs("c19/runs_add_adam_n16.csv", 60000)
x12 = json_runs("c14/runs_xor_adam_n12.csv", 20000)

rows = [("n = 6, addition (C-01)", med(c01[("6", "add")].values())),
        ("n = 8, addition (C-01)", med(c01[("8", "add")].values())),
        ("n = 8, XOR (C-01)", med(c01[("8", "xor")].values())),
        ("n = 10, addition (C-06)", med(c06[("10", "0.5", "add")].values())),
        ("n = 12, addition (C-14, cap 20,000)", med(n12)),
        ("n = 12, XOR (C-14)", med(x12)),
        ("n = 16, addition (C-19, cap 60,000)", med(n16))]
print("Table 1. Median learning time per output bit (steps), bit 0 first")
for name, m in rows: print(f"  {name}: " + " ".join(f"{v:.0f}" for v in m))
for name, runs, cap in (("n = 12", n12, 20000), ("n = 16 (60,000 steps)", n16, 60000)):
    cells = [t for r in runs for t in r]; print(f"  censored cells {name}: {100 * np.mean([t == cap for t in cells]):.1f}%")

print("\nR-2: slowest bit and top/slowest ratio at four accuracy thresholds (n = 8, C-01)")
for thr in (0.90, 0.95, 0.99, 0.999):
    m = med(bit_times("c01/curves.csv", ["n", "op"], thr=thr, cap=10000)[("8", "add")].values())
    print(f"  threshold {thr}: slowest bit {int(np.argmax(m))}, top/slowest {m[-1] / max(m):.2f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
for (name, m), col in zip([rows[1], rows[3], rows[4], rows[6]], ["#1f4e79", "#2e86c1", "#e67e22", "#c0392b"]):
    ax.plot(range(len(m)), m, "o-", color=col, label=name.split(" (")[0].replace(", addition", " addition"))
ax.plot(range(len(rows[2][1])), rows[2][1], "s--", color="#7f8c8d", label="n = 8 XOR")
ax.set_yscale("log"); ax.set_xlabel("output bit (0 = least significant; the last one is the carry out)")
ax.set_ylabel("steps until the bit is learned (median, log scale)")
ax.set_title("Middle bits are learned last; XOR bits all at once")
ax.legend(frameon=False, fontsize=9, ncol=2); fig.tight_layout()
os.makedirs("results", exist_ok=True); fig.savefig("results/part1_learning_order.png", dpi=150)
print("\nfigure: results/part1_learning_order.png")
