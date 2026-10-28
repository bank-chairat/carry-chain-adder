"""Small helpers shared by the summarize_part*.py scripts (README tables and figures)."""
import csv, glob
import numpy as np

def rows(*patterns):
    """All rows of every CSV matching the patterns, in sorted file order."""
    out = []
    for p in patterns:
        files = sorted(glob.glob(p))
        if not files: raise FileNotFoundError(p)
        for f in files: out += list(csv.DictReader(open(f)))
    return out

def t(v, cap):
    """A saved bit time; -1 (never reached within the run) is counted as the cap."""
    v = int(float(v)); return cap if v < 0 else v

def med(x): return float(np.median(x))

def bits(r, cap, n=None):
    """Per-bit times t0, t1, ... of one row."""
    k = sorted((c for c in r if c[0] == "t" and c[1:].isdigit()), key=lambda c: int(c[1:]))
    return [t(r[c], cap) for c in k][:n]

def per_bit_median(rs, cap):
    return [med(col) for col in zip(*[bits(r, cap) for r in rs])]

def fmt(xs): return " ".join(f"{x:,.0f}" for x in xs)

def where(rs, **kw): return [r for r in rs if all(str(r[k]) == str(v) for k, v in kw.items())]
