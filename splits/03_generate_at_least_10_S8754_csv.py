import os
import pandas as pd


dataset = "S8754"
min_mutations = 10

base_dir = os.path.dirname(__file__)
data_root = os.path.join(base_dir, "..", "data")
dataset_dir = os.path.join(data_root, dataset)
out_csv = os.path.join(base_dir, f"{dataset}_ge{min_mutations}.csv")


rows = []


for protein in sorted(os.listdir(dataset_dir)):
    prot_dir = os.path.join(dataset_dir, protein)
    if not os.path.isdir(prot_dir):
        continue
    csv_path = os.path.join(prot_dir, "data.csv")

    df = pd.read_csv(csv_path)

    if len(df) >= min_mutations:
        rows.append([protein, dataset])

pd.DataFrame(rows, columns=["protein_name", "dataset_name"]).to_csv(out_csv, index=False)
print(f"Done: {out_csv}, total {len(rows)} proteins (mutations ≥ {min_mutations})")
