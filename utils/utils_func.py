import torch
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

data_root = Path(__file__).resolve().parents[1] / "data"

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def log_param_grad_norms(model, max_params: int = 50):
    print("\n[Grad norms per parameter - in model order]")
    count = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            grad_str = "grad = None"
        else:
            grad_str = f"grad_norm = {p.grad.data.norm(2).item():.4e}"
        print(f"{name:60s} {grad_str}")
        count += 1
        if count >= max_params:
            print(f"... (only showing first {max_params} params)")
            break


def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [to_gpu(i, device=device) for i in obj]
    elif isinstance(obj, tuple):
        return (to_gpu(i, device=device) for i in obj)
    elif isinstance(obj, dict):
        return {i: to_gpu(j, device=device) for i, j in obj.items()}
    else:
        return obj


def parse_mutation_name(mut: str):

    m = re.match(r"([A-Z\*])(\d+)([A-Z\*])", mut)
    wt_aa, pos_str, mut_aa = m.groups()
    pos = int(pos_str)
    return wt_aa, pos, mut_aa


def get_pred_from_matrix(
    single_var_pred: torch.Tensor,
    mut_list: List[str],
    aa_order: str = AA_ORDER,
) -> torch.Tensor:
    """
    get predictions for a list of mutations from the full prediction matrix.
    Args:
        single_var_pred: Tensor of shape [L, 20], containing per-site AA predictions.
        mut_list: List of mutation names like ["A12G", "D45Y"].
        aa_order: String of length 20, the order of amino acids in the prediction matrix.
    """
    assert single_var_pred.dim() == 2, f"expect [L, 20], got {single_var_pred.shape}"

    L, A = single_var_pred.shape

    preds = []

    for mut in mut_list:
        wt_aa, pos, mut_aa = parse_mutation_name(mut)

        aa_idx = aa_order.index(mut_aa)

        preds.append(single_var_pred[pos, aa_idx])

    preds = torch.stack(preds, dim=0)  # [num_mut]
    return preds


def _create_stratified_protein_splits(mut_csv_path: str, n_splits: int, seed: int):
    """
    Create a 10% hold-out test set and an n-fold multilabel-stratified CV on the remaining 90%.
    Stratification is based on mutation positions parsed from strings (e.g., "A12G,B34C").
    Returns:
      - test_data: (test_mutations, test_labels)
      - cv_folds_data: list of length n_splits with {"train": (...), "val": (...)} per fold.
    """

    df = pd.read_csv(mut_csv_path)

    mutation_names = df["mutation_name"].tolist()

    positions = set()
    for mut in mutation_names:
        for single_mut in str(mut).split(","):
            positions.add(int(single_mut[1:-1]))

    sorted_positions = sorted(list(positions))
    pos_to_idx = {pos: i for i, pos in enumerate(sorted_positions)}

    y = np.zeros((len(mutation_names), len(sorted_positions)), dtype=int)
    for i, mut in enumerate(mutation_names):
        for single_mut in str(mut).split(","):
            y[i, pos_to_idx[int(single_mut[1:-1])]] = 1

    indices = np.arange(df.shape[0])

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    train_val_indices, test_indices = next(msss.split(indices.reshape(-1, 1), y))

    test_df = df.iloc[test_indices]
    test_muts = test_df["mutation_name"].tolist()
    test_labels = torch.as_tensor(test_df["label"].values, dtype=torch.float32)
    test_data = (test_muts, test_labels)

    train_val_df = df.iloc[train_val_indices]
    y_train_val = y[train_val_indices]

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_folds_data = []

    for train_fold_idx, val_fold_idx in mskf.split(train_val_df, y_train_val):
        fold_train_df = train_val_df.iloc[train_fold_idx]
        fold_val_df = train_val_df.iloc[val_fold_idx]

        train_muts = fold_train_df["mutation_name"].tolist()
        train_labels = torch.as_tensor(fold_train_df["label"].values, dtype=torch.float32)

        val_muts = fold_val_df["mutation_name"].tolist()
        val_labels = torch.as_tensor(fold_val_df["label"].values, dtype=torch.float32)

        cv_folds_data.append({"train": (train_muts, train_labels), "val": (val_muts, val_labels)})

    return test_data, cv_folds_data


def build_all_lookup_tables(protein_df, n_splits=5, seed=42):
    """
    Precompute data splits for each protein and return:
      - test_lookup: {protein_name -> (test_mutations, test_labels)}  # 10% hold-out
      - cv_lookup_data: {
          protein_name -> [
            {
              "train": (train_mutations, train_labels),
              "val":   (val_mutations,   val_labels)
            }
          ] * n_splits
        }
    """

    test_lookup = {}
    cv_lookup_data = {}

    print("Pre-computing all data splits for all proteins...")
    for name in protein_df.index:
        dataset_name = protein_df.loc[name, "dataset_name"]
        mut_csv_path = f"{data_root}/{dataset_name}/{name}/data.csv"

        # This core function is called only once per protein
        test_data, cv_folds = _create_stratified_protein_splits(mut_csv_path, n_splits=n_splits, seed=seed)

        if test_data and cv_folds:
            test_lookup[name] = test_data
            cv_lookup_data[name] = cv_folds

    print("Pre-computation finished.")
    return test_lookup, cv_lookup_data


def build_uncertainty_lookups_from_residuals(
    protein_df,
    fit_test_lookup,
    fit_cv_lookup_data,
    residual_csv: str = "residual_label.csv",
    residual_column: str = "residual",
):
    """
    Build uncertainty-model lookups that reuse the exact same splits
    as the fitness model, but:
      - keep only single mutants
      - use residuals from residual_label.csv as labels.

    Parameters
    protein_df : pd.DataFrame
        Must have index = protein_name and a column "dataset_name".
    fit_test_lookup : dict
        {protein_name -> (test_mutations, test_labels)} from build_all_lookup_tables().
    fit_cv_lookup_data : dict
        {protein_name -> [ {"train": (...), "val": (...)} * n_splits ]}.
    residual_csv : str
        File name under each protein folder (e.g., "residual_label.csv").
    residual_column : str
        Column name in residual csv that stores the residual label.

    Returns
    -------
    uncertainty_test_lookup : dict
        {protein_name -> (test_single_muts, test_residuals)}.
    uncertainty_cv_lookup_data : dict
        {protein_name -> [ {"train": (...), "val": (...)} * n_splits ]} but
        only with single mutants and residual labels.
    """

    def is_single_mut(mut_name: str) -> bool:
        return "," not in str(mut_name)

    uncertainty_test_lookup = {}
    uncertainty_cv_lookup_data = {}

    print("Building uncertainty lookups (single mutants + residuals)...")
    data_root_path = Path(data_root)
    residual_filename = residual_csv
    for name in protein_df.index:
        dataset_name = protein_df.loc[name, "dataset_name"]
        residual_path = data_root_path / dataset_name / name / residual_filename

        # Load residual_label.csv for this protein
        resid_df = pd.read_csv(residual_path)

        # Map mutation_name -> residual (assumed single mutants only)
        resid_map = dict(
            zip(
                resid_df["mutation_name"].astype(str),
                resid_df[residual_column].astype(float),
            )
        )

        #  Build test lookup for uncertainty model
        fit_test_muts, _ = fit_test_lookup[name]

        uncertainty_test_muts = []
        uncertainty_test_residuals = []

        for m in fit_test_muts:
            m_str = str(m)
            if not is_single_mut(m_str):
                # skip double/multi mutants
                continue
            uncertainty_test_muts.append(m_str)
            uncertainty_test_residuals.append(resid_map[m_str])

        uncertainty_test_residuals = torch.as_tensor(uncertainty_test_residuals, dtype=torch.float32)
        uncertainty_test_lookup[name] = (uncertainty_test_muts, uncertainty_test_residuals)
        #  Build CV folds for uncertainty model
        fit_folds = fit_cv_lookup_data[name]
        uncertainty_folds = []

        # fit_folds: list of {"train": (...), "val": (...)}*5
        # "train": (train_mutations, train_labels), type(train_mutations) = List[str], type(train_labels) = Tensor
        for fold in fit_folds:
            fit_train_muts, _ = fold["train"]
            fit_val_muts, _ = fold["val"]

            # Train set: keep only single mutants with residuals
            uncertainty_train_muts = []
            uncertainty_train_residuals = []
            for m in fit_train_muts:
                m_str = str(m)
                if not is_single_mut(m_str):
                    continue
                uncertainty_train_muts.append(m_str)
                uncertainty_train_residuals.append(resid_map[m_str])

            # Val set: same filtering
            uncertainty_val_muts = []
            uncertainty_val_residuals = []
            for m in fit_val_muts:
                m_str = str(m)
                if not is_single_mut(m_str):
                    continue
                uncertainty_val_muts.append(m_str)
                uncertainty_val_residuals.append(resid_map[m_str])

            uncertainty_train_residuals = torch.as_tensor(uncertainty_train_residuals, dtype=torch.float32)
            uncertainty_val_residuals = torch.as_tensor(uncertainty_val_residuals, dtype=torch.float32)
            uncertainty_folds.append(
                {
                    "train": (uncertainty_train_muts, uncertainty_train_residuals),
                    "val": (uncertainty_val_muts, uncertainty_val_residuals),
                }
            )

        uncertainty_cv_lookup_data[name] = uncertainty_folds

    print("Confidence lookups finished.")
    return uncertainty_test_lookup, uncertainty_cv_lookup_data


def zscore_within_protein(labels):
    labels = torch.tensor(labels, dtype=torch.float32)
    return (labels - labels.mean()) / (labels.std(unbiased=False) + 1e-8)


def read_wt_idx_from_fasta(fasta_path: str) -> torch.LongTensor:
    """Read the first FASTA record and return wt_idx as LongTensor[L] (AA20)."""
    seq = []
    for line in Path(fasta_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq:  # stop after the first record
                break
            continue
        seq.append(line.upper())

    if not seq:
        raise ValueError(f"No sequence found in FASTA: {fasta_path}")

    wt_seq = "".join(seq)
    try:
        wt_idx = torch.tensor([AA_TO_IDX[aa] for aa in wt_seq], dtype=torch.long)
    except KeyError as e:
        raise ValueError(f"Found non-AA20 residue: {e.args[0]}") from None

    return wt_idx