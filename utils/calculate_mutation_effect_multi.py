import torch
import pandas as pd
import numpy as np
import math

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


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
            wt = mut[0]
            mt = mut[-1]
            pos = int(mut[1:-1])
            data.append({"sample_id": i, "pos": pos, "wt": AA_TO_IDX[wt], "mt": AA_TO_IDX[mt]})
    return pd.DataFrame(data)


def df_to_tensor(df, max_mut=10, device="cpu"):
    """
    Convert a mutation DataFrame to a padded tensor.

    Args:
        df: DataFrame with columns: sample_id, pos, wt, mt
        max_mut: Maximum number of mutations per sample
        device: Torch device

    Returns:
        tensor: LongTensor of shape [B, max_mut, 3], where the last dimension is (pos, wt, mt)
        mask: BoolTensor of shape [B, max_mut], True for valid mutations and False for padding
    """
    df = df.copy()
    sample_ids, inverse = np.unique(df["sample_id"].values, return_inverse=True)
    df["mapped_id"] = inverse
    df["mut_idx"] = df.groupby("mapped_id").cumcount()

    B = len(sample_ids)
    tensor = torch.full((B, max_mut, 3), -1, dtype=torch.long, device=device)

    idx_sample = torch.tensor(df["mapped_id"].values, dtype=torch.long, device=device)
    idx_mut = torch.tensor(df["mut_idx"].values, dtype=torch.long, device=device)
    pos = torch.tensor(df["pos"].values, dtype=torch.long, device=device)
    wt = torch.tensor(df["wt"].values, dtype=torch.long, device=device)
    mt = torch.tensor(df["mt"].values, dtype=torch.long, device=device)

    tensor[idx_sample, idx_mut, 0] = pos
    tensor[idx_sample, idx_mut, 1] = wt
    tensor[idx_sample, idx_mut, 2] = mt

    mask = tensor[:, :, 0] != -1
    return tensor, mask


def high_order_product_ge2(
    U: torch.Tensor,                 # [L, 20, r]
    mutations_tensor: torch.Tensor, # [B, M, 3]
    mask: torch.Tensor,             # [B, M] bool
    eps: float = 1e-6,
    normalize_by_pairs: bool = False,
) -> torch.Tensor:
    """
    Compute the >=2-order contribution using a generating-function expansion.

    For each rank channel:
        prod(1 + u) = 1 + e1 + e2 + e3 + ...
    where e1 is the first-order sum, e2 is the pairwise elementary symmetric sum, etc.

    This function returns:
        ge2 = prod(1 + u) - 1 - e1 = e2 + e3 + ...

    Args:
        U: Tensor of shape [L, 20, r]
        mutations_tensor: Tensor of shape [B, M, 3]
        mask: Bool tensor of shape [B, M]
        eps: Small positive constant for numerical stability
        normalize_by_pairs: If True, divide the result by C(m, 2)

    Returns:
        body: FloatTensor of shape [B]
    """
    pos = mutations_tensor[..., 0]
    mt = mutations_tensor[..., 2]

    pos_safe = pos.clone()
    mt_safe = mt.clone()
    pos_safe[~mask] = 0
    mt_safe[~mask] = 0

    u = U[pos_safe, mt_safe]
    u = u * mask.unsqueeze(-1).to(u.dtype)

    # Enforce 1 + u > 0 before log1p
    u = torch.clamp(u, min=-1.0 + eps)

    log_prod = torch.log1p(u).sum(dim=1)   # [B, r]
    prod = torch.exp(log_prod)             # [B, r]

    first_order = u.sum(dim=1)             # [B, r]
    ge2_r = prod - 1.0 - first_order       # [B, r]

    m = mask.sum(dim=1)
    ge2_r = ge2_r * (m >= 2).unsqueeze(-1).to(ge2_r.dtype)

    body = ge2_r.sum(dim=-1) / math.sqrt(U.shape[-1])

    if normalize_by_pairs:
        m_float = m.to(body.dtype)
        denom = (m_float * (m_float - 1) / 2.0).clamp(min=1.0)
        body = body / denom

    return body


def calculate_batch_prediction(
    single_mut_matrix,   # [L, 20] 
    U,                   # [L, 20, r] 
    mut_name_list,       # list[str] 
    max_mut=44,
    device="cpu",
):
    """
    Compute batch predictions:
        pred = single_sum + ge2_body

    where ge2_body is the >=2-order interaction contribution obtained from
    the generating-function expansion.

    Args:
        single_mut_matrix: Tensor of shape [L, 20]
        mut_name_list: List of mutation strings
        U: Tensor of shape [L, 20, r]
        max_mut: Maximum number of mutations per sample
        device: Torch device

    Returns:
        preds: FloatTensor of shape [B]
        If return_cache is True, also returns cache
    """
    df = build_mutation_table(mut_name_list)
    mutations_tensor, mask = df_to_tensor(df, max_mut=max_mut, device=device)

    pos = mutations_tensor[..., 0]
    mt = mutations_tensor[..., 2]

    pos_safe = pos.clone()
    mt_safe = mt.clone()
    pos_safe[~mask] = 0
    mt_safe[~mask] = 0

    single_vals = single_mut_matrix[pos_safe, mt_safe]
    single_sum = (single_vals * mask.to(single_vals.dtype)).sum(dim=1)

    ge2_body = high_order_product_ge2(
        U,
        mutations_tensor,
        mask,
        normalize_by_pairs=False,
    )

    preds = (single_sum + ge2_body).to(torch.float32)

    return preds