"""README Part 5: what speeds the adder up? Loss weights, width and learning rate, intermediate targets and inputs,
pre-training, and a transformer. Rebuilds Tables 5 to 7 and Figure 5 from saved results (C-20 to C-57, C-99).
Run from the experiments folder: python3 summarize_part5.py"""
import csv, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summary_utils import rows, med, where, per_bit_median, fmt, t, bits

def slowest(rs, cap=10000, n=9): return med([max(bits(r, cap, n)) for r in rs])
def json_runs(path, cap): return [[cap if x < 0 else x for x in json.loads(r["bit_times_json"])] for r in csv.DictReader(open(path))]
def censored(runs, cap): c = [x for r in runs for x in r]; return 100 * np.mean([x == cap for x in c])

print("Loss weights (slowest-bit time, median; n = 8 has a 10,000-step cap)")
w20 = rows("c20/weights_n8.csv"); print(f"  C-20 hand profile [1,1,2,4,8,4,2,1,1]: {slowest(where(w20, w=1)):,.0f} vs uniform {slowest(where(w20, w=0)):,.0f}")
print(f"  C-22 loss-proportional (seeds 0-4): {slowest(rows('c22/adapt_n8.csv')):,.0f}")
c24 = rows("c24/ctrl_n8.csv")
print("  C-24 seeds 5-9: " + ", ".join(f"{k} {slowest(where(c24, w=k)):,.0f}" for k in ("uniform", "adaptive", "shuffled")))
print(f"  C-26 sharper rule (seeds 0-9): {slowest(rows('c26/sharp_n8.csv')):,.0f}")
c38 = rows("c38/lr_n8.csv"); print("  C-38 lr 3e-3: " + ", ".join(f"{k} {slowest(where(c38, w=k)):,.0f}" for k in ("uniform", "adaptive")))
for name, u, a, cap in (("C-21/C-23 n = 12", "c21/runs_add_adam_n12_uniform.csv", "c23/runs_add_adam_n12_adaptive.csv", 20000),
                        ("C-19/C-27 n = 16, 60k steps", "c19/runs_add_adam_n16.csv", "c27/runs_add_adam_n16_adaptive60k.csv", 60000),
                        ("C-41/C-42 n = 16, lr 3e-3", "c41/runs_add_adam_n16_lr3e-3.csv", "c42/runs_add_adam_n16_adaptive_lr3e-3.csv", 20000)):
    U, A = json_runs(u, cap), json_runs(a, cap)
    print(f"  {name}: median of per-seed slowest bit: uniform {med([max(r) for r in U]):,.0f} ({censored(U, cap):.1f}% cells censored), adaptive {med([max(r) for r in A]):,.0f} ({censored(A, cap):.1f}%);"
          f" slowest per-bit median: uniform {max(med(c) for c in zip(*U)):,.0f}, adaptive {max(med(c) for c in zip(*A)):,.0f}")

print("\nTable 5. Width and learning rate (C-39 grid, slowest-bit time, median of 5 seeds)")
grid = {(W, lr): slowest(rows(f"c39_W{W}_lr{lr}/wide_n8.csv")) for W in (128, 256, 512) for lr in ("1e-3", "3e-3", "1e-2")}
for W in (128, 256, 512): print(f"  W = {W}: " + "  ".join(f"lr {lr}: {grid[(W, lr)]:,.0f}" for lr in ("1e-3", "3e-3", "1e-2")))
X = np.array([[1, np.log10(float(lr)), np.log10(W)] for (W, lr) in grid]); y = np.log10(list(grid.values()))
c = np.linalg.lstsq(X, y, rcond=None)[0]; print(f"  fit log10 T = {c[0]:.2f} + {c[1]:.2f} log10(lr) + {c[2]:.2f} log10(W); largest residual {np.max(np.abs(X @ c - y)):.3f}")
c35 = rows("c35/wide_n8.csv"); print("  C-35 width 64..1024 at lr 1e-3: " + ", ".join(f"{W}: {slowest(where(c35, k=W)):,.0f}" for W in (64, 128, 256, 512, 1024)))
print(f"  C-36 W = 1024 at lr 3e-4: {slowest(rows('c36c/wide_n8.csv')):,.0f}")
for name, f in (("C-40 n = 12 lr 3e-3", "c40/runs_add_adam_n12_lr3e-3.csv"), ("C-40 n = 12 lr 1e-2", "c40/runs_add_adam_n12_lr1e-2.csv"), ("C-41 n = 16 lr 3e-3", "c41/runs_add_adam_n16_lr3e-3.csv")):
    R = json_runs(f, 20000); print(f"  {name}: slowest bit {med([max(r) for r in R]):,.0f}, censored {censored(R, 20000):.1f}%")

print("\nTable 6. Intermediate targets and inputs (n = 8; slowest bit / bit 4, median of 5 seeds)")
a43, a44, i49 = rows("c43/aux_n8.csv"), rows("c44/aux4_n8.csv"), rows("c49/inputs_n8.csv")
d2, z0, pre = rows("c99_deg2/inputs_n8.csv"), rows("c99_zeros/inputs_n8.csv"), rows("c55/pretrain_n8.csv")
cases = [("no help", where(a43, cond="none")), ("+ carries as extra outputs", where(a43, cond="carry")),
         ("+ propagates as extra outputs", where(a43, cond="prop")), ("+ random parities as extra outputs", where(a43, cond="rand")),
         ("propagates as extra inputs", where(i49, cond="pin")), ("random parities as extra inputs", where(i49, cond="randin")),
         ("random pair XORs as extra inputs", d2), ("zero columns as extra inputs", z0),
         ("500 steps of XOR pre-training", where(pre, cond="xor")), ("500 steps of random-parity pre-training", where(pre, cond="rand"))]
for name, R in cases: print(f"  {name:40s} {slowest(R):6,.0f} / {med([t(r['t4'], 10000) for r in R]):,.0f}")
print("  C-44 bit 4 with one extra output: " + ", ".join(f"{k} {med([t(r['t4'], 10000) for r in where(a44, cond=k)]):,.0f}" for k in ("p4", "c4", "p4c4", "pnot4", "cnot4")))
a45 = rows("c45/pi_n8.csv"); print("  C-45 bit i with p_i as extra output: " + ", ".join(f"bit {i} {med([t(r[f't{i}'], 10000) for r in where(a45, cond=f'p{i}')]):,.0f}" for i in (1, 3, 5, 6, 7)))
a48 = rows("c48/aux12_n12.csv"); print("  C-48 n = 12 bit 4: " + ", ".join(f"{k} {med([t(r['time_bit4'], 20000) for r in where(a48, cond=k)]):,.0f}" for k in ("none", "p4", "c4", "p4c4")))
for name, f in (("C-50 n = 12 with propagate inputs", "c50/runs_add_adam_n12_pin.csv"), ("C-50 n = 16 with propagate inputs", "c50/runs_add_adam_n16_pin.csv"), ("C-52 n = 12 with lookahead inputs", "c52/runs_add_adam_n12_lookahead.csv")):
    R = json_runs(f, 20000); print(f"  {name}: per-bit " + fmt([med(c) for c in zip(*R)]) + f"; slowest {med([max(r) for r in R]):,.0f}")
for pre_len in ("100", "2000"):
    P = rows(f"c56_pre{pre_len}/pretrain_n8.csv"); print(f"  C-56 pre-training {pre_len} steps: XOR {slowest(where(P, cond='xor')):,.0f}, random {slowest(where(P, cond='rand')):,.0f}")

print("\nTable 7. Small transformer (C-54 / C-57), per-bit medians, cap 20,000")
print("  plain tokens     : " + fmt(per_bit_median(rows("c54/transformer_n8.csv"), 20000)))
print("  + propagate tokens: " + fmt(per_bit_median(rows("c57/transformer_p_n8.csv"), 20000)))

print("\nThe weak-correlation idea (C-46, C-51, C-53)")
b46 = rows("c46/bias_n8.csv")
for q in ("0.5", "0.35", "0.2"): print(f"  C-46 q = {q}, each bit alone: " + fmt([med([int(r["time_to_099"]) for r in b46 if r["q"] == q and r["bit"] == str(b)]) for b in range(9)]))
p51 = rows("c51/parity.csv"); print("  C-51 two-bit parity, r = " + ", ".join(f"{r}: {med([int(x['time_to_099']) for x in p51 if x['r'] == r]):.0f}" for r in ("0.5", "0.4", "0.3", "0.2", "0.1", "0.05")))
l53 = rows("c53/latent.csv"); print("  C-53 latent threshold, k = " + ", ".join(f"{k}: {med([int(x['time_to_099']) for x in l53 if x['r'] == k]):.0f}" for k in ("7", "6", "5", "4")))

labels = [n for n, _ in cases]; vals = [slowest(R) for _, R in cases]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
cols = ["#7f8c8d", "#2e86c1", "#2e86c1", "#c0392b", "#1f4e79", "#c0392b", "#c0392b", "#7f8c8d", "#2e86c1", "#e67e22"]
ax.barh(range(len(vals))[::-1], vals, color=cols)
ax.set_yticks(range(len(vals))[::-1], labels, fontsize=9); ax.set_xlabel("steps until every sum bit is learned (median of 5 seeds)")
for i, v in zip(range(len(vals))[::-1], vals): ax.text(v + 60, i, f"{v:,.0f}", va="center", fontsize=8.5)
ax.set_title("Only features made from a_i XOR b_i help (n = 8)"); ax.set_xlim(0, max(vals) * 1.15)
fig.tight_layout(); os.makedirs("results", exist_ok=True); fig.savefig("results/part5_interventions.png", dpi=150)
print("\nfigure: results/part5_interventions.png")
