import torch
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm

from calculate_mutation_effect_multi import calculate_batch_prediction
from utils_func import to_gpu, zscore_within_protein,read_wt_idx_from_fasta

data_root = Path(__file__).resolve().parents[1] / "data"


def compute_spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
   
    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    corr, _ = spearmanr(pred_np, target_np)

    return float(corr)


class ProcessingData(torch.utils.data.Dataset):
    def __init__(self, train_csv: pd.DataFrame):
        self.df = train_csv

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        name = self.df.index[index]
        dataset_name = self.df.loc[name, "dataset_name"]
        prefix = data_root / dataset_name / name

        embedding = torch.load(prefix / "embedding_ESM2_650M.pt")
        epistasis_dict = torch.load(prefix / "epistasis_ESM2_650M.pt")
        atom14_coords = torch.load(prefix / "atom14_coords_ESMFold.pt")
        wt_idx = read_wt_idx_from_fasta(prefix / "wt.fasta")

        epistasis = epistasis_dict["epistasis_sym"]
        single_mutation_effects = epistasis_dict["single_mutation_effects"]

        label_df = pd.read_csv(prefix / "data.csv")
        mut_list = label_df["mutation_name"].tolist()
        mut_labels = zscore_within_protein(label_df["label"].to_numpy())

        data = {
            "embedding": embedding,
            "epistasis": epistasis,
            "atom14_coords": atom14_coords,
            "single_mutation_effects": single_mutation_effects,
            "wt_idx": wt_idx,
        }

        return data, name, mut_list, mut_labels


def train_model(model, optimizer, loader, max_mut: int = 44):
    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0.0
    epoch_corr = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch_idx, batch in enumerate(pbar):
        data, name, mut_list, mut_labels = batch

        data = to_gpu(data, device)
        mut_labels = torch.as_tensor(mut_labels, dtype=torch.float32, device=device)

        optimizer.zero_grad(set_to_none=True)

        single_pred, high_delta = model(data)
        pred = calculate_batch_prediction(
            single_pred,
            high_delta,
            mut_list,
            max_mut=max_mut,
            device=device,
        ).to(torch.float32)

        loss = torch.nn.functional.mse_loss(pred, mut_labels)
        corr = compute_spearman(pred, mut_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        epoch_loss += float(loss.detach())
        epoch_corr += float(corr)

        avg_loss = epoch_loss / (batch_idx + 1)
        avg_corr = epoch_corr / (batch_idx + 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", corr=f"{avg_corr:.4f}")

    return epoch_loss / len(loader), epoch_corr / len(loader)


def test_benchmark(model, loader, benchmark_name=None, max_mut: int = 30):
    model.eval()
    device = next(model.parameters()).device
    epoch_corr = 0.0

    desc = f"Test {benchmark_name}" if benchmark_name is not None else "Test"
    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            data, name, mut_list, mut_labels = batch

            data = to_gpu(data, device)
            mut_labels = torch.as_tensor(mut_labels, dtype=torch.float32, device=device)

            single_pred, double_pred = model(data)
            pred = calculate_batch_prediction(
                single_pred,
                double_pred,
                mut_list,
                max_mut=max_mut,
                device=device,
            ).to(torch.float32)

            corr = compute_spearman(pred, mut_labels)
            epoch_corr += corr

            avg_corr = epoch_corr / (batch_idx + 1)
            pbar.set_postfix(corr=f"{avg_corr:.4f}")

    return epoch_corr / len(loader)