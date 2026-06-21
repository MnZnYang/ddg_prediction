# $$\hat{y} = \sum s_i + \sum g(u_i)  +[g(\sum u_i)-\sum g(u_i)]$$

import torch

import pandas as pd
import numpy as np
import math

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


class EpistasisMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        """
        Args:
            input_dim: The dimension 'r' of your U vectors (e.g., 256).
            hidden_dim: Hidden layer size for the MLP.
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )  # Helps with training stability  # Non-linear activation  # Output a single scalar score

    def forward(self, h_combined):
        """
        Args:
            h_combined: [Batch, r] - The sum of all mutation embeddings for each sample.
        Returns:
            interaction_score: [Batch]
        """
        return self.net(h_combined).squeeze(-1)


def build_mutation_table(mutation_lists):
    """
    Args:
        mutation_lists: list[str], e.g. ["A1G", "A5M,Y10D", ...] (0-based positions)
    Returns:
        DataFrame with columns: sample_id, pos, wt, mt
    """
    data = []
    for i, s in enumerate(mutation_lists):
        for mut in s.split(","):
            if not mut:
                continue  # Handle empty strings if any
            wt = mut[0]
            mt = mut[-1]
            try:
                pos = int(mut[1:-1])
            except ValueError:
                continue  # Skip malformed strings
            data.append({"sample_id": i, "pos": pos, "wt": AA_TO_IDX[wt], "mt": AA_TO_IDX[mt]})
    return pd.DataFrame(data)


def df_to_tensor(df, max_mut=10, device="cpu"):
    """
    Convert mutation DataFrame to a padded tensor.
    """
    if df.empty:
        # Handle empty case safely
        return torch.zeros((0, max_mut, 3), dtype=torch.long, device=device), torch.zeros((0, max_mut), dtype=torch.bool, device=device)

    df = df.copy()
    sample_ids, inverse = np.unique(df["sample_id"].values, return_inverse=True)
    df["mapped_id"] = inverse
    df["mut_idx"] = df.groupby("mapped_id").cumcount()

    B = len(sample_ids)
    # If B is smaller than expected (some samples filtered out), we might need handling,
    # but usually mutation_lists len determines B.
    # Here we rely on input list length which might be safer in full implementation.
    # For this snippet, we use valid samples count.

    tensor = torch.full((B, max_mut, 3), -1, dtype=torch.long, device=device)

    idx_sample = torch.tensor(df["mapped_id"].values, dtype=torch.long, device=device)
    idx_mut = torch.tensor(df["mut_idx"].values, dtype=torch.long, device=device)
    pos = torch.tensor(df["pos"].values, dtype=torch.long, device=device)
    wt = torch.tensor(df["wt"].values, dtype=torch.long, device=device)
    mt = torch.tensor(df["mt"].values, dtype=torch.long, device=device)

    # Safe indexing
    mask_valid = idx_mut < max_mut
    tensor[idx_sample[mask_valid], idx_mut[mask_valid], 0] = pos[mask_valid]
    tensor[idx_sample[mask_valid], idx_mut[mask_valid], 1] = wt[mask_valid]
    tensor[idx_sample[mask_valid], idx_mut[mask_valid], 2] = mt[mask_valid]

    mask = tensor[:, :, 0] != -1
    return tensor, mask


def calculate_batch_prediction_mlp(
    single_mut_matrix,  # [L, 20]
    U,  # [L, 20, r]
    mut_name_list,  # List[str]
    mlp_model,
    max_mut=10,
    device="cpu",
):
    df = build_mutation_table(mut_name_list)
    mutations_tensor, mask = df_to_tensor(df, max_mut=max_mut, device=device)

    pos = mutations_tensor[..., 0]
    mt = mutations_tensor[..., 2]

    pos_safe = pos.clone()
    mt_safe = mt.clone()
    pos_safe[~mask] = 0
    mt_safe[~mask] = 0

    u_vecs = U[pos_safe, mt_safe]  # [B, M, r]
    u_vecs = u_vecs * mask.unsqueeze(-1).to(u_vecs.dtype)

    h_combined = u_vecs.sum(dim=1)

    interaction_score = mlp_model(h_combined)  # [B]

    single_vals = single_mut_matrix[pos_safe, mt_safe]  # [B, M]
    single_sum = (single_vals * mask.to(single_vals.dtype)).sum(dim=1)  # [B]

    preds = single_sum + interaction_score  # [B]
    return preds.to(torch.float32)
