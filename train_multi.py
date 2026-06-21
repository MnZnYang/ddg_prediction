import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, "splits")
utils_dir = os.path.join(base_dir, "utils")

sys.path.insert(0, base_dir)
sys.path.insert(0, utils_dir)

from model_l20r.SE3_Transformer.SE3_Transformer_v6 import SE3Transformer
from utils_for_ddG_prediction import ProcessingData, train_model, test_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SE3Transformer on all training data and evaluate on benchmarks."
    )

    parser.add_argument("--train_csv", type=str, default=os.path.join(csv_dir, "training_data.csv"))
    parser.add_argument("--s669_csv", type=str, default=os.path.join(csv_dir, "S669_ge10.csv"))
    parser.add_argument("--s461_csv", type=str, default=os.path.join(csv_dir, "S461_ge10.csv"))
    # parser.add_argument("--s8754_csv", type=str, default=os.path.join(csv_dir, "S8754_ge10.csv"))
    
    parser.add_argument("--chitosanase_csv", type=str, default=os.path.join(csv_dir, "Chitosanase.csv"))

    parser.add_argument("--out_dir", type=str, default=os.path.join(base_dir, "output_64"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_mut_train", type=int, default=44)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--hidden_dim_0", type=int, default=320)
    parser.add_argument("--hidden_dim_1", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--rankH", type=int, default=64)
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
        hidden_fiber_dict={0: args.hidden_dim_0, 1: args.hidden_dim_1},
        out_fiber_dict={0: 128, 1: 32},
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
    # s8754_df = pd.read_csv(args.s8754_csv, index_col=0)
    chitosanase_df = pd.read_csv(args.chitosanase_csv, index_col=0)

    print(f"Total training proteins: {len(train_df)}", flush=True)
    print(f"S669 proteins: {len(s669_df)}", flush=True)
    print(f"S461 proteins: {len(s461_df)}", flush=True)
    # print(f"S8754 proteins: {len(s8754_df)}", flush=True)
    print(f"Chitosanase proteins: {len(chitosanase_df)}", flush=True)

    train_loader = make_loader(
        train_df,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    s669_loader = make_loader(
        s669_df,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    s461_loader = make_loader(
        s461_df,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    # s8754_loader = make_loader(
    #     s8754_df,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     seed=args.seed,
    # )
    chitosanase_loader = make_loader(
        chitosanase_df,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    best_s669 = float("-inf")
    best_s461 = float("-inf")
    # best_s8754 = float("-inf")
    best_chitosanase = float("-inf")

    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_corr = train_model(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            max_mut=args.max_mut_train,
        )

        s669_corr = test_benchmark(model, s669_loader, "S669")
        s461_corr = test_benchmark(model, s461_loader, "S461")
        # s8754_corr = test_benchmark(model, s8754_loader, "S8754")
        chitosanase_corr = test_benchmark(model, chitosanase_loader, "Chitosanase")

        current_lr = optimizer.param_groups[0]["lr"]

        best_s669 = max(best_s669, s669_corr) if np.isfinite(s669_corr) else best_s669
        best_s461 = max(best_s461, s461_corr) if np.isfinite(s461_corr) else best_s461
        # best_s8754 = max(best_s8754, s8754_corr) if np.isfinite(s8754_corr) else best_s8754
        best_chitosanase = max(best_chitosanase, chitosanase_corr) if np.isfinite(chitosanase_corr) else best_chitosanase

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_corr": train_corr,
                "test_S669_corr": s669_corr,
                "test_S461_corr": s461_corr,
                # "test_S8754_corr": s8754_corr,
                "test_Chitosanase_corr": chitosanase_corr,
                "lr": current_lr,
            }
        )

        pd.DataFrame(log_rows).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)

        print(
            f"Epoch {epoch:04d} | "
            f"lr={current_lr:.6e} | "
            f"train_loss={train_loss:.6f} | "
            f"train_corr={train_corr:.4f} | "
            f"S669={s669_corr:.4f} | "
            f"S461={s461_corr:.4f} | "
            # f"S8754={s8754_corr:.4f} | "
            f"Chitosanase={chitosanase_corr:.4f}",
            flush=True,
        )

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))

    summary = pd.DataFrame(
        [
            {
                "best_S669_corr": best_s669,
                "best_S461_corr": best_s461,
                # "best_S8754_corr": best_s8754,
                "best_Chitosanase_corr": best_chitosanase,
                "last_train_loss": log_rows[-1]["train_loss"] if log_rows else np.nan,
                "last_train_corr": log_rows[-1]["train_corr"] if log_rows else np.nan,
            }
        ]
    )
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    print("Training finished.", flush=True)
    print(f"Results saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()