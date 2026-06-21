from pathlib import Path

import torch
import pandas as pd
import os

from metrics import spearman_corr
from utils_func import log_param_grad_norms, to_gpu, get_pred_from_matrix


data_root = Path(__file__).resolve().parents[1] / "data"


class ProcessingData(torch.utils.data.Dataset):

    def __init__(self, train_csv):
        self.df = train_csv

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        name = self.df.index[index]
        dataset_name = self.df.loc[name, "dataset_name"]
        prefix = f"{data_root}/{dataset_name}/{name}"

        embedding = torch.load(f"{prefix}/embedding_ESM2_650M.pt")
        epistasis = torch.load(f"{prefix}/epistasis_ESM2_650M.pt")
        ca_coords = torch.load(f"{prefix}/CA_coords_ESMFold.pt")
        atom14_coords = torch.load(f"{prefix}/atom14_coords_ESMFold.pt")
        epistasis_sym = epistasis["epistasis_sym"]
        single_mutation_effects = epistasis["single_mutation_effects"]
        wt_fasta_path = f"{prefix}/wt.fasta"
        with open(wt_fasta_path, "r") as f:
            wt_lines = [line.strip() for line in f if not line.startswith(">")]
        wt_seq = "".join(wt_lines)
        data = {
            "embedding": embedding,
            "epistasis": epistasis_sym,
            "ca_coords": ca_coords,
            "atom14_coords": atom14_coords,
            "wt_seq": wt_seq,
            "single_mutation_effects": single_mutation_effects,
        }

        return data, name


def train_model(model, optimizer, loader, train_lookup):
    """
    Train the confidence head with MSE loss on |residual| and
    track five metrics on the training set:

      1) mean_mse: per-protein MSE(pred, |error|), averaged over proteins
      2) mean_spearman: Spearman(pred, |error|) per protein, averaged
      3) mean_mae_0_10: MAE of the 0–10% most confident mutations
                        (smallest pred → highest confidence)
      4) mean_mae_10_20: MAE of the 10–20% most confident mutations
      5) mean_prec_0_10: Precision@Top-10%:
                         among the 10% most confident mutations
                         (smallest pred), fraction that also lie in
                         the 10% smallest |error|
    """
    model.train()
    device = next(model.parameters()).device

    sum_mse = 0.0
    sum_spearman = 0.0
    sum_mae_0_10 = 0.0
    sum_mae_10_20 = 0.0
    sum_prec_0_10 = 0.0

    for batch_idx, (data, name) in enumerate(loader):
        name = name[0]  # batch_size=1
        mut_list, errors = train_lookup[name]

        data = to_gpu(data, device)
        errors = errors.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        pred_error_matrix = model(data)
        pred_error = get_pred_from_matrix(pred_error_matrix, mut_list).to(torch.float32)

        abs_err = errors.abs()

        loss = torch.nn.functional.mse_loss(pred_error, abs_err)  # scalar
        with torch.no_grad():
            # (1) per-protein mean MSE
            sum_mse += loss.item()

            # (2) Spearman(pred_error, |error|)
            corr = spearman_corr(pred_error, abs_err).item()
            sum_spearman += corr

            # (3) & (4) MAE on 0–10% and 10–20% most confident mutations
            # smaller pred_error => higher confidence
            sorted_idx = torch.argsort(pred_error, dim=0)  # [N_mut]
            n_mut = sorted_idx.numel()

            k10 = int(0.10 * n_mut)
            k20 = int(0.20 * n_mut)

            idx_0_10 = sorted_idx[:k10]
            mae_0_10 = abs_err[idx_0_10].mean().item()
            sum_mae_0_10 += mae_0_10

            idx_10_20 = sorted_idx[k10:k20]
            mae_10_20 = abs_err[idx_10_20].mean().item()
            sum_mae_10_20 += mae_10_20

            # (5) Precision@Top-10%:
            err_sorted_idx = torch.argsort(abs_err, dim=0)
            idx_good_0_10 = err_sorted_idx[:k10]

            conf_mask = torch.zeros(n_mut, dtype=torch.bool, device=device)
            good_mask = torch.zeros(n_mut, dtype=torch.bool, device=device)
            conf_mask[idx_0_10] = True
            good_mask[idx_good_0_10] = True

            n_conf = conf_mask.sum().item()
            n_both = (conf_mask & good_mask).sum().item()
            prec_0_10 = n_both / n_conf
            sum_prec_0_10 += prec_0_10

        loss.backward()

        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        if batch_idx == 0:
            print(f"\n[train_confidence_model] global grad norm before clipping: {total_grad_norm:.4e}")
            log_param_grad_norms(model, max_params=800)

        optimizer.step()

    num_proteins = len(loader)
    mean_mse = sum_mse / num_proteins
    mean_spearman = sum_spearman / num_proteins
    mean_mae_0_10 = sum_mae_0_10 / num_proteins
    mean_mae_10_20 = sum_mae_10_20 / num_proteins
    mean_prec_0_10 = sum_prec_0_10 / num_proteins

    return mean_mse, mean_spearman, mean_mae_0_10, mean_mae_10_20, mean_prec_0_10


def validation_model(model, loader, validation_lookup):
    """
    Same metrics as train_model, but no backward / optimizer step.
    """
    model.eval()
    device = next(model.parameters()).device

    sum_mse = 0.0
    sum_spearman = 0.0
    sum_mae_0_10 = 0.0
    sum_mae_10_20 = 0.0
    sum_prec_0_10 = 0.0

    with torch.no_grad():
        for data, name in loader:
            name = name[0]

            mut_list, errors = validation_lookup[name]
            data = to_gpu(data, device)
            errors = errors.to(device).float()
            abs_err = errors.abs()

            pred_error_matrix = model(data)
            pred_error = get_pred_from_matrix(pred_error_matrix, mut_list).to(torch.float32)  # [N_mut]

            # 1) per-protein mean MSE
            loss = torch.nn.functional.mse_loss(pred_error, abs_err)
            sum_mse += loss.item()

            # 2) Spearman(pred_error, |error|)
            corr = spearman_corr(pred_error, abs_err)
            sum_spearman += corr.item()

            # 3 & 4) MAE on 0–10% and 10–20% most confident mutations
            sorted_idx = torch.argsort(pred_error, dim=0)
            n_mut = sorted_idx.numel()

            k10 = int(0.10 * n_mut)
            k20 = int(0.20 * n_mut)

            idx_0_10 = sorted_idx[:k10]
            idx_10_20 = sorted_idx[k10:k20]

            mae_0_10 = abs_err[idx_0_10].mean().item()
            mae_10_20 = abs_err[idx_10_20].mean().item()

            sum_mae_0_10 += mae_0_10
            sum_mae_10_20 += mae_10_20

            # 5) Precision@Top-10%
            err_sorted_idx = torch.argsort(abs_err, dim=0)
            idx_good_0_10 = err_sorted_idx[:k10]

            conf_mask = torch.zeros(n_mut, dtype=torch.bool, device=device)
            good_mask = torch.zeros(n_mut, dtype=torch.bool, device=device)
            conf_mask[idx_0_10] = True
            good_mask[idx_good_0_10] = True

            n_conf = conf_mask.sum().item()
            n_both = (conf_mask & good_mask).sum().item()
            prec_0_10 = n_both / n_conf
            sum_prec_0_10 += prec_0_10

    num_proteins = len(loader)
    mean_mse = sum_mse / num_proteins
    mean_spearman = sum_spearman / num_proteins
    mean_mae_0_10 = sum_mae_0_10 / num_proteins
    mean_mae_10_20 = sum_mae_10_20 / num_proteins
    mean_prec_0_10 = sum_prec_0_10 / num_proteins

    return mean_mse, mean_spearman, mean_mae_0_10, mean_mae_10_20, mean_prec_0_10


def test_model(model, loader, test_lookup):
    """
    Test the uncertainty model on held-out residuals.

    Metrics:
      - per protein: MSE(pred, |error|), Spearman(pred, |error|),
                     MAE on 0–10% and 10–20% most confident mutations
      - global: average of the above metrics over all proteins

    Returns:
      mean_mse, mean_spearman, mean_mae_0_10, mean_mae_10_20, mean_prec_0_10,
      per_protein_metrics

    where per_protein_metrics is:
      { protein_name: (mse, spearman, mae_0_10, mae_10_20, prec_0_10) }
    """
    model.eval()
    device = next(model.parameters()).device

    sum_mse = 0.0
    sum_spearman = 0.0
    sum_mae_0_10 = 0.0
    sum_mae_10_20 = 0.0
    sum_prec_0_10 = 0.0

    per_protein_metrics = {}

    with torch.no_grad():
        for data, name in loader:
            name = name[0]

            mut_list, errors = test_lookup[name]
            data = to_gpu(data, device)
            errors = errors.to(device).float()
            abs_errors = errors.abs()

            pred_error_matrix = model(data)
            pred_error = get_pred_from_matrix(pred_error_matrix, mut_list).to(torch.float32)  # [N_mut]

            # 1) per-protein mean MSE(pred, |error|)
            loss = torch.nn.functional.mse_loss(pred_error, abs_errors)
            mse_protein = loss.item()

            # 2) per-protein Spearman(pred, |error|)
            spearman = spearman_corr(pred_error, abs_errors).item()
            # 3) & 4) MAE on 0–10% and 10–20% most confident mutations
            sorted_idx = torch.argsort(pred_error, dim=0)
            n_mut = sorted_idx.numel()
            k10 = int(0.10 * n_mut)
            k20 = int(0.20 * n_mut)

            idx_0_10 = sorted_idx[:k10]
            mae_0_10 = abs_errors[idx_0_10].mean().item()

            idx_10_20 = sorted_idx[k10:k20]
            mae_10_20 = abs_errors[idx_10_20].mean().item()

            # 5) Precision@Top-10%
            err_sorted_idx = torch.argsort(abs_errors, dim=0)
            idx_good_0_10 = err_sorted_idx[:k10]

            conf_mask = torch.zeros(n_mut, dtype=torch.bool, device=device)
            good_mask = torch.zeros(n_mut, dtype=torch.bool, device=device)
            conf_mask[idx_0_10] = True
            good_mask[idx_good_0_10] = True

            n_conf = conf_mask.sum().item()
            n_both = (conf_mask & good_mask).sum().item()
            prec_0_10 = n_both / n_conf

            # save per-protein metrics
            per_protein_metrics[name] = (mse_protein, spearman, mae_0_10, mae_10_20, prec_0_10)

            # accumulate global means
            sum_mse += mse_protein
            sum_spearman += spearman
            sum_mae_0_10 += mae_0_10
            sum_mae_10_20 += mae_10_20
            sum_prec_0_10 += prec_0_10

    num_proteins = len(loader)
    mean_mse = sum_mse / num_proteins
    mean_spearman = sum_spearman / num_proteins
    mean_mae_0_10 = sum_mae_0_10 / num_proteins
    mean_mae_10_20 = sum_mae_10_20 / num_proteins
    mean_prec_0_10 = sum_prec_0_10 / num_proteins

    return mean_mse, mean_spearman, mean_mae_0_10, mean_mae_10_20, mean_prec_0_10, per_protein_metrics


def test_benchmark(model, loader, benchmark_name: str):
    """
    Evaluate on an external benchmark (e.g. S461), where residual_label.csv provides residuals.

    Metrics:
      - mean_spearman: Spearman(pred_error, |residual|) averaged over proteins
      - mean_mse: per-protein MSE(pred_error, |residual|) averaged
      - mean_prec_top3: Precision@Top-3 averaged over proteins
      - mean_prec_top5: Precision@Top-5 averaged over proteins

    """
    model.eval()
    device = next(model.parameters()).device

    sum_spearman = 0.0
    sum_mse = 0.0
    sum_prec_top3 = 0.0
    sum_prec_top5 = 0.0

    with torch.no_grad():
        for data, name in loader:
            name = name[0]

            csv_path = os.path.join(data_root, benchmark_name, name, "residual_label.csv")
            df = pd.read_csv(csv_path)
            mut_list = df["mutation_name"].tolist()

            residual = torch.tensor(
                df["residual"].to_numpy(),
                dtype=torch.float32,
                device=device,
            )
            abs_residual = residual.abs()

            data = to_gpu(data, device)
            pred_error_matrix = model(data)
            pred_error = get_pred_from_matrix(pred_error_matrix, mut_list).to(torch.float32)

            # (1) Spearman(pred_error, |residual|)
            corr = spearman_corr(pred_error, abs_residual)
            sum_spearman += corr.item()

            # (2) Per-protein MSE(pred_error, |residual|)
            mse_scalar = torch.nn.functional.mse_loss(pred_error, abs_residual)
            sum_mse += mse_scalar.item()

            # (3) Precision@Top-3  (4) Precision@Top-5
            n_mut = pred_error.numel()

            sorted_idx_conf = torch.argsort(pred_error, dim=0)
            sorted_idx_true = torch.argsort(abs_residual, dim=0)

            top3_conf_idx = sorted_idx_conf[:3]
            top3_true_idx = sorted_idx_true[:3]

            conf_mask_3 = torch.zeros(n_mut, dtype=torch.bool, device=device)
            true_mask_3 = torch.zeros(n_mut, dtype=torch.bool, device=device)
            conf_mask_3[top3_conf_idx] = True
            true_mask_3[top3_true_idx] = True

            n_both_3 = (conf_mask_3 & true_mask_3).sum().item()
            prec_top3 = n_both_3 / 3.0
            sum_prec_top3 += prec_top3

            top5_conf_idx = sorted_idx_conf[:5]
            top5_true_idx = sorted_idx_true[:5]

            conf_mask_5 = torch.zeros(n_mut, dtype=torch.bool, device=device)
            true_mask_5 = torch.zeros(n_mut, dtype=torch.bool, device=device)
            conf_mask_5[top5_conf_idx] = True
            true_mask_5[top5_true_idx] = True

            n_both_5 = (conf_mask_5 & true_mask_5).sum().item()
            prec_top5 = n_both_5 / 5.0
            sum_prec_top5 += prec_top5

    num_proteins = len(loader)

    mean_spearman = sum_spearman / num_proteins
    mean_mse = sum_mse / num_proteins
    mean_prec_top3 = sum_prec_top3 / num_proteins
    mean_prec_top5 = sum_prec_top5 / num_proteins

    return mean_spearman, mean_mse, mean_prec_top3, mean_prec_top5
