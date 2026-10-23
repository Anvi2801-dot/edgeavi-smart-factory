import glob
import os

import pandas as pd

files = sorted(glob.glob("data/ground_truth_*.csv"), key=os.path.getmtime, reverse=True)
if not files:
    raise SystemExit("No ground_truth_*.csv found - run the simulation first")
print("reading", files[0])
d = pd.read_csv(files[0])
pred = d.engine_status == "Defective"
gt = d.gt_defective == 1
tp, tn = (gt & pred).sum(), (~gt & ~pred).sum()
fp, fn = (~gt & pred).sum(), (gt & ~pred).sum()
print(f"n={len(d)}  TP {tp}, TN {tn}, FP {fp}, FN {fn}, accuracy {100 * (tp + tn) / len(d):.1f}%")