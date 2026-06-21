import os
import pandas as pd

base_dir = os.path.dirname(__file__)
data_root = os.path.abspath(os.path.join(base_dir, "..", "data"))
out_csv = os.path.join(base_dir, "training_data.csv")

rows = []


dataset1 = "cdna_ddG_data_transitivity"
dir1 = os.path.join(data_root, dataset1)
for name in sorted(os.listdir(dir1)):
    p = os.path.join(dir1, name)
    if not os.path.isdir(p):
        continue
    rows.append([name, dataset1])


dataset2 = "humandomain_sup2_data"
dir2 = os.path.join(data_root, dataset2)
for name in sorted(os.listdir(dir2)):
    p = os.path.join(dir2, name)
    if not os.path.isdir(p):
        continue
    rows.append([name, dataset2])


dataset3 = "ArchStabMS_1E10"
dir3 = os.path.join(data_root, dataset3)
for name in sorted(os.listdir(dir3)):
    p = os.path.join(dir3, name)
    if not os.path.isdir(p):
        continue
    rows.append([name, dataset3])

pd.DataFrame(rows, columns=["protein_name", "dataset_name"]).to_csv(out_csv, index=False, encoding="utf-8")
print(f"Done: {out_csv}, total {len(rows)} proteins (sup2 kept as-is)")
