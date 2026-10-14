#!/usr/bin/env python3
"""
kappa.py -- inter-coder agreement for the design census.

1. python kappa.py make     -> writes coding_sheet.csv: a random 30-row sample of the census
                                with the name, bib key and platform only (no classification).
2. A second coder fills the columns 'regime' (mw|1w|fw) and 'components' (space-separated
   subset of GEMM ATT KV SM NORM ROPE ACT MOE) using the rule of Sec. "Classification rule
   and robustness", from the primary sources only.
3. python kappa.py score    -> Cohen's kappa for the regime, and per-component kappa
                                (presence/absence) against census_data.py.
"""
import csv, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from census_data import CENSUS

H = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(H, "coding_sheet.csv")
COMPS = ["GEMM", "ATT", "KV", "SM", "NORM", "ROPE", "ACT", "MOE"]

def kappa(a, b):
    labels = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

if sys.argv[1:] == ["make"]:
    random.seed(2026)
    rows = random.sample(CENSUS, 30)
    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name", "bibkey", "year", "platform", "regime", "components"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], "", ""])
    print("wrote", SHEET)
elif sys.argv[1:] == ["score"]:
    ref = {r[1]: r for r in CENSUS}
    got = [r for r in csv.DictReader(open(SHEET, encoding="utf-8")) if r["regime"].strip()]
    a = [ref[g["bibkey"]][4] for g in got]; b = [g["regime"].strip() for g in got]
    print(f"n = {len(got)}  regime kappa = {kappa(a, b):.2f}  raw agreement = "
          f"{sum(x == y for x, y in zip(a, b)) / len(a):.0%}")
    for c in COMPS:
        a = [c in ref[g["bibkey"]][6] for g in got]; b = [c in g["components"].split() for g in got]
        print(f"  {c:5s} kappa = {kappa(a, b):.2f}")
else:
    print(__doc__)
