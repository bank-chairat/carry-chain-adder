import csv, json, glob, numpy as np
for f in sorted(glob.glob("c14/runs_*.csv")):
    R = list(csv.DictReader(open(f)))
    if not R: continue
    n = int(R[0]["n"]); T = np.array([json.loads(r["bit_times_json"]) for r in R], float)
    cens = (T < 0); T[cens] = 20000
    med = np.median(T, 0).astype(int); slow = int(np.argmax(med))
    print(f.split('/')[-1], "seeds", len(R), "median", med.tolist(), "slowest bit", slow, "mid(n/4..3n/4)", n/4 <= slow <= 3*n/4,
          "top/slow", round(med[-1]/med[slow], 2), "censored frac", round(cens[:, :n+1].mean(), 3), "per-bit cens", cens.sum(0).tolist())
    if "xor" in f: print("   frac <=2000:", round((T[:, :n] <= 2000).mean(), 3))
