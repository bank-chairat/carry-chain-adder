"""C-174 (check): recompute the numbers quoted for C-169 to C-173 from the raw outputs."""
import csv, numpy as np
def rows(d, f="confine.csv"): return list(csv.DictReader(open(f"{d}/{f}")))
def rng(v, k=4): return f"{min(v):.{k}f}-{max(v):.{k}f}"
for d in ["c169_ln", "c170_rms", "c170_center", "c171_fixnorm", "c173_adagrad_ln"]:
    R = rows(d); out = []
    for arm in ("alone", "conf3", "joint"):
        t = [int(r["t4"]) if int(r["t4"]) > 0 else 10000 for r in R if r["arm"] == arm]
        if t: out.append(f"{arm} {t} median {np.median(t):.0f}")
    j = [r for r in R if r["arm"] == "joint"]; al = np.median([int(r["t4"]) for r in R if r["arm"] == "alone"]); jm = np.median([int(r["t4"]) if int(r["t4"]) > 0 else 10000 for r in j])
    extra = f"; corr {rng([float(r['L2norm_corrS250']) for r in j])}" if "L2norm_corrS250" in j[0] else ""
    print(f"{d}: " + "; ".join(out) + f"; joint/alone {jm / al:.2f}; joint R2 L2 {rng([float(r['L2_R2S250']) for r in j])}, L3 {rng([float(r['L3_R2S250']) for r in j])}{extra}")
for d in ["c170_none", "c170_ln"]:
    j = rows(d); print(f"{d}: corr(norm, S) {rng([float(r['L2norm_corrS250']) for r in j])}")
R = rows("c170_n12ln", "n12confine.csv")
for arm in ("alone", "joint"): t = [int(r["time_bit4"]) for r in R if r["arm"] == arm]; print(f"c170_n12ln {arm} {t} median {np.median(t):.0f}")
for d in ("none", "ln"):
    B = rows(f"c172_{d}", "bits.csv"); print(f"c172 {d} medians per bit", [float(np.median([int(r[f't{i}']) for r in B])) for i in range(9)])
