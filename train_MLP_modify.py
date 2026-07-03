import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, "splits")
utils_dir = os.path.join(base_dir, "utils")

sys.path.insert(0, base_dir)
sys.path.insert(0, utils_dir)

from model_l20r.SE3_Transformer.SE3_Transformer_v6 import SE3Transformer
from utils.calculate_mutation_effect_MLP import EpistasisMLP, calculate_batch_prediction_mlp
from utils.utils_for_ddG_prediction_MLP import ProcessingData, test_benchmark, unwrap_mut_list
from utils.utils_func import to_gpu
from loss.spearman_loss.loss import SpearmanLoss


def compute_spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    corr, _ = spearmanr(pred_np, target_np)
    return float(corr)


def pairwise_margin_loss(preds: torch.Tensor, targets: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    preds_i = preds.unsqueeze(1)  # [N, 1]
    preds_j = preds.unsqueeze(0)  # [1, N]
    targets_i = targets.unsqueeze(1)
    targets_j = targets.unsqueeze(0)

    mask = targets_i > targets_j  # [N, N]

    if not mask.any():
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    p_i = preds_i.expand_as(mask)[mask]
    p_j = preds_j.expand_as(mask)[mask]

    y = torch.ones_like(p_i)

    loss_fn = torch.nn.MarginRankingLoss(margin=margin)
    return loss_fn(p_i, p_j, y)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SE3Transformer with Step-based Evaluation and Margin Loss.")

    parser.add_argument("--train_csv", type=str, default=os.path.join(csv_dir, "training_data.csv"))
    parser.add_argument("--s669_csv", type=str, default=os.path.join(csv_dir, "S669_ge10.csv"))
    parser.add_argument("--s461_csv", type=str, default=os.path.join(csv_dir, "S461_ge10.csv"))
    parser.add_argument("--chitosanase_csv", type=str, default=os.path.join(csv_dir, "Chitosanase.csv"))

    parser.add_argument("--out_dir", type=str, default=os.path.join(base_dir, "output_MLP/stage1"))
    parser.add_argument("--base_model_path", type=str, default=os.path.join(base_dir, "output_MLP/stage1/best_S461.pt"))

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=200)

    parser.add_argument("--margin_weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_mut_train", type=int, default=44)
    parser.add_argument("--max_mut_test", type=int, default=44)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--hidden_dim_0", type=int, default=320)
    parser.add_argument("--hidden_dim_1", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--rankH", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)

    parser.add_argument("--geo_neighbor", type=float, default=1 / 3)
    parser.add_argument("--epi_neighbor", type=float, default=1 / 3)

    return parser.parse_args()


def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args) -> SE3Transformer:
    model = SE3Transformer(
        heads=args.heads,
        dim_head=args.dim_head,
        depth=args.depth,
        hidden_fiber_dict={
            0: args.hidden_dim_0,
            1: args.hidden_dim_1,
        },
        out_fiber_dict={
            0: 128,
            1: 32,
        },
        rankH=args.rankH,
        geo_neighbor=args.geo_neighbor,
        epi_neighbor=args.epi_neighbor,
    )
    return model


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    dataset = ProcessingData(df)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        collate_fn=(lambda batch: batch[0]) if batch_size == 1 else None,
    )
    return loader


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed_everywhere(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    train_df = pd.read_csv(args.train_csv, index_col=0)
    s669_df = pd.read_csv(args.s669_csv, index_col=0)
    s461_df = pd.read_csv(args.s461_csv, index_col=0)
    chitosanase_df = pd.read_csv(args.chitosanase_csv, index_col=0)

    print(f"Total training proteins: {len(train_df)}", flush=True)

    train_loader = make_loader(train_df, args.batch_size, True, args.num_workers, args.seed)
    s669_loader = make_loader(s669_df, 1, False, args.num_workers, args.seed)
    s461_loader = make_loader(s461_df, 1, False, args.num_workers, args.seed)
    chitosanase_loader = make_loader(chitosanase_df, 1, False, args.num_workers, args.seed)

    model = build_model(args).to(device)
    mlp_model = EpistasisMLP(input_dim=args.rankH, hidden_dim=args.mlp_hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mlp_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.min_lr)

    best_s669 = float("-inf")
    best_s461 = float("-inf")
    best_chitosanase = float("-inf")

    log_rows = []

    global_step = 0
    batch_counter = 0
    train_iter = iter(train_loader)

    model.train()
    mlp_model.train()
    optimizer.zero_grad(set_to_none=True)

    epoch_mse_loss = 0.0
    epoch_rank_loss = 0.0
    epoch_corr = 0.0

    pbar = tqdm(total=args.max_steps, desc="Global Steps")

    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        data, name, mut_list, mut_labels = batch
        mut_list = unwrap_mut_list(mut_list)
        data = to_gpu(data, device)
        mut_labels = torch.as_tensor(mut_labels, dtype=torch.float32, device=device).flatten()

        single_pred, U = model(data)
        pred = calculate_batch_prediction_mlp(
            single_mut_matrix=single_pred,
            U=U,
            mut_name_list=mut_list,
            mlp_model=mlp_model,
            max_mut=args.max_mut_train,
            device=device,
        )

        mse_loss = torch.nn.functional.mse_loss(pred, mut_labels)
        rank_loss = pairwise_margin_loss(pred, mut_labels, margin=args.margin)
        total_loss = mse_loss + args.margin_weight * rank_loss

        corr = compute_spearman(pred, mut_labels)

        (total_loss / args.accum_steps).backward()

        epoch_mse_loss += float(mse_loss.detach())
        epoch_rank_loss += float(rank_loss.detach())
        epoch_corr += float(corr)
        batch_counter += 1

        if batch_counter % args.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(mlp_model.parameters()), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.update(1)

            avg_mse = epoch_mse_loss / batch_counter
            avg_rank = epoch_rank_loss / batch_counter
            avg_corr = epoch_corr / batch_counter
            pbar.set_postfix(MSE=f"{avg_mse:.4f}", RankL=f"{avg_rank:.4f}", corr=f"{avg_corr:.4f}")

            if global_step % args.eval_steps == 0:
                current_train_loss = avg_mse + args.margin_weight * avg_rank
                current_train_corr = avg_corr

                # 重置累加器
                epoch_mse_loss = 0.0
                epoch_rank_loss = 0.0
                epoch_corr = 0.0
                batch_counter = 0

                s669_corr = test_benchmark(model, mlp_model, s669_loader, "S669", args.max_mut_test)
                s461_corr = test_benchmark(model, mlp_model, s461_loader, "S461", args.max_mut_test)
                chitosanase_corr = test_benchmark(model, mlp_model, chitosanase_loader, "Chitosanase", args.max_mut_test)

                current_lr = optimizer.param_groups[0]["lr"]

                if np.isfinite(s669_corr):
                    best_s669 = max(best_s669, s669_corr)
                if np.isfinite(s461_corr):
                    best_s461 = max(best_s461, s461_corr)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "mlp_model": mlp_model.state_dict(),
                            "global_step": global_step,
                            "s461_corr": best_s461,
                        },
                        os.path.join(args.out_dir, "best_S461.pt"),
                    )
                    print(f"  >>> [Checkpoint] New Best S461: {best_s461:.4f} saved")
                if np.isfinite(chitosanase_corr):
                    best_chitosanase = max(best_chitosanase, chitosanase_corr)

                log_rows.append(
                    {
                        "step": global_step,
                        "train_loss": current_train_loss,
                        "train_mse": avg_mse,
                        "train_rank_loss": avg_rank,
                        "train_corr": current_train_corr,
                        "test_S669_corr": s669_corr,
                        "test_S461_corr": s461_corr,
                        "test_Chitosanase_corr": chitosanase_corr,
                        "best_S669_corr": best_s669,
                        "best_S461_corr": best_s461,
                        "best_Chitosanase_corr": best_chitosanase,
                        "lr": current_lr,
                    }
                )

                pd.DataFrame(log_rows).to_csv(os.path.join(args.out_dir, "training_step_log.csv"), index=False)

                print(
                    f"\nStep {global_step:05d} | lr={current_lr:.6e} | MSE={avg_mse:.4f} | RankL={avg_rank:.4f} | train_corr={current_train_corr:.4f}\n" f"-> S669={s669_corr:.4f} | S461={s461_corr:.4f} | Chito={chitosanase_corr:.4f}",
                    flush=True,
                )

                torch.save(
                    {
                        "model": model.state_dict(),
                        "mlp_model": mlp_model.state_dict(),
                        "global_step": global_step,
                        "args": vars(args),
                    },
                    os.path.join(args.out_dir, "last.pt"),
                )

                model.train()
                mlp_model.train()

    print("Training finished.", flush=True)
    print(f"Results saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
