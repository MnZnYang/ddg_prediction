import torch
import soft_rank_pytorch
from metrics import _rank_data


def NLL_loss(residual: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    residual : y_true - y_pred_main, shape [N]
    log_var   : predicted log σ^2, shape [N]

    Returns: NLL for each sample, shape [N]
    NLL = 0.5 * (log σ^2 + e^2 / σ^2)  (ignoring constant 0.5*log(2π))
    """

    log_var = torch.clamp(log_var, min=-10.0, max=10.0)

    inv_var = torch.exp(-log_var)
    nll = 0.5 * (log_var + (residual**2) * inv_var)
    return nll


def spearman_loss(pred, true, regularization_strength, regularization):

    assert pred.device == true.device
    assert pred.shape == true.shape
    assert pred.shape[0] == 1
    assert pred.ndim == 2

    device = pred.device

    soft_pred = soft_rank_pytorch.soft_rank(pred.cpu(), regularization_strength=regularization_strength, regularization=regularization).to(device)
    soft_true = _rank_data(true.squeeze(0)).to(device)
    preds_diff = soft_pred - soft_pred.mean()
    target_diff = soft_true - soft_true.mean()

    cov = (preds_diff * target_diff).mean()
    preds_std = torch.sqrt((preds_diff * preds_diff).mean())
    target_std = torch.sqrt((target_diff * target_diff).mean())

    spearman_corr = cov / (preds_std * target_std + 1e-6)
    return -spearman_corr


def pearson_loss(pred, true):

    assert pred.shape == true.shape
    assert pred.ndim == 1

    preds_diff = pred - pred.mean()
    target_diff = true - true.mean()

    cov = (preds_diff * target_diff).mean()
    preds_std = torch.sqrt((preds_diff * preds_diff).mean())
    target_std = torch.sqrt((target_diff * target_diff).mean())

    return -cov / (preds_std * target_std + 1e-6)
