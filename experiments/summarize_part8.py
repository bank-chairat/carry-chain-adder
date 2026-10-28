"""README Part 8: LayerNorm removes the delay; which part of it matters, and what it says about the hump.
Rebuilds Tables 9 and 10 and Figure 8 from saved results (C-169 to C-178, C-184 to C-192).
Run from the experiments folder: python3 summarize_part8.py"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summary_utils import rows, med, where, t, fmt, per_bit_median

def arm(f, a, cap=10000, col="t4"):
    R = [r for r in rows(f) if r.get("arm") == a]; return med([t(r[col], cap) for r in R]), R
def rng(R, col): v = [float(r[col]) for r in R]; return f"{min(v):.3f} to {max(v):.3f}"

base_alone = arm("c116_W128_FW0/confine.csv", "alone")[0]
print("Table 9. Bit-4 time with normalisation or read-out changes (n = 8, width 128, Adam 1e-3; median of 5 seeds)")
cases = [("no normalisation (C-116)", "c116_W128_FW0/confine.csv", "joint"), ("LayerNorm in every hidden layer (C-169)", "c169_ln/confine.csv", "joint"),
         ("divide by RMS only (C-170)", "c170_rms/confine.csv", "joint"), ("subtract mean only (C-170)", "c170_center/confine.csv", "joint"),
         ("LayerNorm in layers 1-2 only (C-176)", "c176_ln12/confine.csv", "joint"), ("LayerNorm in layer 3 only (C-176)", "c176_ln3/confine.csv", "joint"),
         ("weight rows held at their initial norm (C-171)", "c171_fixnorm/confine.csv", "joint"),
         ("only the OTHER heads read RMS-normalised layer 3 (C-177)", "c177/confine.csv", "joint_orms"),
         ("only bit 4's head reads RMS-normalised layer 3 (C-177)", "c177/confine.csv", "joint_4rms"),
         ("bit 4's head reads mean-centred layer 3 (C-178)", "c178/confine.csv", "joint_4center"),
         ("as C-177 but RMS constant in the backward pass (C-188)", "c188/confine.csv", "joint_4rms_sg"),
         ("bit 4's head divides by the fixed S-profile of the size (C-189)", "c189/confine.csv", "joint_4divS"),
         ("bit 4's head divides by one number per batch (C-190)", "c190/confine.csv", "joint_4glob"),
         ("bit 4's read-out step size x1 (C-192 control)", "c192/confine.csv", "joint_4lr1.0"),
         ("bit 4's read-out step size x0.25 (C-192)", "c192/confine.csv", "joint_4lr0.25"),
         ("bit 4's read-out step size x0.1 (C-192)", "c192/confine.csv", "joint_4lr0.1")]
fig_rows = []
for name, f, a in cases:
    j, R = arm(f, a); al = arm(f, "alone")[0] if any(r.get("arm") == "alone" for r in rows(f)) else base_alone
    extra = f"; layer-2 R^2 on S at 250 {rng(R, 'L2_R2S250')}" + (f"; corr(size of layer-2 vector, S) {rng(R, 'L2norm_corrS250')}" if "L2norm_corrS250" in R[0] else "")
    print(f"  {name:66s} {j:6,.0f}  ratio {j / al:5.1f}{extra}")
    fig_rows.append((name, j / al))
j, _ = arm("c187/confine.csv", "joint_4mag"); al = arm("c187/confine.csv", "alone")[0]
print(f"  {'LayerNorm net, bit 4 reads layer 3 times the fixed S-profile (C-187)':66s} {j:6,.0f}  ratio {j / al:5.1f}")
print(f"  (alone_4mag: {arm('c187/confine.csv', 'alone_4mag')[0]:,.0f}, LayerNorm joint {arm('c187/confine.csv', 'joint')[0]:,.0f})")

print("\nLayerNorm in other settings (joint/alone)")
for name, f, cap in (("Adagrad + LayerNorm (C-173)", "c173_adagrad_ln", 10000), ("width 64 (C-185)", "c185_W64", 10000), ("lr 3e-4 (C-185)", "c185_lr3e-4", 20000), ("frozen random layer (C-185)", "c185_FW1024", 10000)):
    a_, j_ = arm(f"{f}/confine.csv", "alone", cap)[0], arm(f"{f}/confine.csv", "joint", cap)[0]; print(f"  {name:30s} alone {a_:,.0f} joint {j_:,.0f} ratio {j_ / a_:.2f}")
R = rows("c170_n12ln/n12confine.csv"); a_, j_ = med([t(r["time_bit4"], 20000) for r in where(R, arm="alone")]), med([t(r["time_bit4"], 20000) for r in where(R, arm="joint")])
print(f"  {'n = 12 with LayerNorm (C-170)':30s} alone {a_:,.0f} joint {j_:,.0f} ratio {j_ / a_:.2f}")

print("\nTable 10. Every sum bit, with and without LayerNorm (median of 5 seeds)")
b8 = {k: per_bit_median(rows(f"c172_{k}/bits.csv"), 10000) for k in ("none", "ln")}
b12 = {k: per_bit_median(rows(f"c175_{k}/bits12.csv"), 20000) for k in ("none", "ln")}
for k in ("none", "ln"): print(f"  n = 8,  {k:4s}: " + fmt(b8[k]))
for k in ("none", "ln"): print(f"  n = 12, {k:4s}: " + fmt(b12[k]))
for nm, f, cap, nb in (("n = 8", "c172", 10000, 9), ("n = 12", "c175", 20000, 13)):
    for k in ("none", "ln"):
        R = rows(f"{f}_{k}/bits{'12' if nb == 13 else ''}.csv"); print(f"  {nm} {k}: slowest bit per seed " + fmt([max(t(r[f't{i}'], cap) for i in range(nb)) for r in R]))
s184 = rows("c184_ln/bits.csv"); single = [med([int(r["t"]) for r in s184 if r["bit"] == str(b)]) for b in range(9)]
print("  C-184 each bit alone with LayerNorm: " + fmt(single))

print("\nC-191 (from c191_scale.txt): RMS of the last hidden layer during training")
for line in open("c191_scale.txt"): print("  " + line.rstrip())

fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
for k, col, lab in (("none", "#c0392b", "no normalisation"), ("ln", "#1f4e79", "LayerNorm")):
    axs[0].plot(range(9), b8[k], "o-", color=col, label=f"n = 8, {lab}"); axs[0].plot(range(13), b12[k], "s--", color=col, alpha=0.7, label=f"n = 12, {lab}")
axs[0].set_yscale("log"); axs[0].set_xlabel("sum bit"); axs[0].set_ylabel("steps until learned (median, log)")
axs[0].set_title("(a) LayerNorm flattens the hump (C-172, C-175)", fontsize=10); axs[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
short = ["no normalisation", "LayerNorm, all layers", "RMS scaling only", "mean-centring only", "LayerNorm layers 1-2", "LayerNorm layer 3",
         "fixed-norm weight rows", "others' heads read RMS-normalised", "bit 4's head reads RMS-normalised", "bit 4's head reads centred",
         "same, RMS constant backward", "bit 4 divides by S-profile", "bit 4 divides by batch RMS", "read-out step x1", "read-out step x0.25", "read-out step x0.1"]
y = np.arange(len(fig_rows))[::-1]; v = [r[1] for r in fig_rows]
axs[1].barh(y, v, color=["#1f4e79" if x < 3 else "#c0392b" for x in v]); axs[1].set_yticks(y, short, fontsize=8)
axs[1].set_xscale("log"); axs[1].axvline(1, color="k", lw=0.6); axs[1].set_xlabel("bit-4 time / time alone (log; blue = below 3)")
axs[1].set_title("(b) Which part of the normalisation matters", fontsize=10); fig.tight_layout()
os.makedirs("results", exist_ok=True); fig.savefig("results/part8_layernorm.png", dpi=150)
print("\nfigure: results/part8_layernorm.png")
