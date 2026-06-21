import os
import pandas as pd

base_dir = os.path.dirname(__file__)
data_root = os.path.abspath(os.path.join(base_dir, "..", "data"))
out_csv = os.path.join(base_dir, "Chitosanase.csv")

rows = []


dataset = "Chitosanase_Tm"
dir1 = os.path.join(data_root, dataset)
for name in sorted(os.listdir(dir1)):
    p = os.path.join(dir1, name)
    if not os.path.isdir(p):
        continue
    rows.append([name, dataset])


pd.DataFrame(rows, columns=["protein_name", "dataset_name"]).to_csv(out_csv, index=False, encoding="utf-8")
print(f"Done: {out_csv}, total {len(rows)} proteins (sup2 kept as-is)")
