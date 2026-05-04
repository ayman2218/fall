import argparse
import os
from typing import Iterable

import numpy as np


# MediaPipe Pose (33 landmarks) connections (undirected)
# Source: standard landmark topology (face omitted; pose only)
# We'll use a conservative subset that covers main limbs + torso.
MEDIAPIPE_POSE_EDGES: list[tuple[int, int]] = [
    # Torso
    (11, 12),  # shoulders
    (11, 23), (12, 24),  # shoulders to hips
    (23, 24),  # hips
    # Left arm
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # Head/neck-ish
    (0, 1), (0, 4),
    (1, 2), (2, 3),
    (4, 5), (5, 6),
    (9, 10),
    (0, 11), (0, 12),
    (7, 9), (8, 10),
]


def build_adjacency(n_joints: int, edges: Iterable[tuple[int, int]]) -> np.ndarray:
    A = np.zeros((n_joints, n_joints), dtype=np.float32)
    for i, j in edges:
        if i >= n_joints or j >= n_joints:
            continue
        A[i, j] = 1.0
        A[j, i] = 1.0
    # self-loops
    np.fill_diagonal(A, 1.0)
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    # D^{-1/2} A D^{-1/2}
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.empty_like(deg, dtype=np.float32)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    D = np.diag(deg_inv_sqrt.astype(np.float32))
    return (D @ A @ D).astype(np.float32)


def compute_class_weight(y: np.ndarray) -> dict[int, float] | None:
    y = y.astype(np.int32).reshape(-1)
    u, c = np.unique(y, return_counts=True)
    counts = {int(k): int(v) for k, v in zip(u, c)}
    n0 = float(counts.get(0, 0))
    n1 = float(counts.get(1, 0))
    print(f"Train label counts: {counts} (0=normal, 1=fall)")
    if n0 <= 0 or n1 <= 0:
        return None
    total = n0 + n1
    w = {0: total / (2.0 * n0), 1: total / (2.0 * n1)}
    print(f"Using class_weight: {w}")
    return w


def build_model(*, n_channels: int, A_norm: np.ndarray):
    import torch
    import torch.nn as nn

    A = torch.tensor(A_norm, dtype=torch.float32)

    class GraphConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B,C,T,V)
            x = torch.einsum("vw,bctw->bctv", A.to(x.device), x)
            return self.proj(x)

    class STGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn1 = GraphConv(n_channels, 64)
            self.bn1 = nn.BatchNorm2d(64)
            self.tcn1 = nn.Conv2d(64, 64, kernel_size=(9, 1), padding=(4, 0))
            self.drop1 = nn.Dropout(0.3)

            self.gcn2 = GraphConv(64, 128)
            self.bn2 = nn.BatchNorm2d(128)
            self.tcn2 = nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0))
            self.drop2 = nn.Dropout(0.4)

            self.head = nn.Linear(128, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B,T,V,C) -> (B,C,T,V)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.gcn1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = torch.relu(self.tcn1(x))
            x = self.drop1(x)

            x = self.gcn2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = torch.relu(self.tcn2(x))
            x = self.drop2(x)

            # global average pool over T and V
            x = x.mean(dim=2).mean(dim=2)  # (B,128)
            logits = self.head(x).squeeze(-1)  # (B,)
            return logits

    return STGCN()


def main() -> None:
    p = argparse.ArgumentParser(description="Train a simple ST-GCN for fall detection from gcn_sequences.npz")
    p.add_argument(
        "--data-npz",
        default=r"C:\Users\p\Pictures\data_organised_images_224\gcn_sequences.npz",
        help="Produced by prepare_gcn_sequences.py",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip training and only evaluate using an existing checkpoint in --out-model.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--out-model",
        default=r"C:\Users\p\Pictures\data_organised_images_224\gcn_fall_model.pt",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for fall class (1). Lower => higher recall, more false positives.",
    )
    p.add_argument(
        "--auto-threshold",
        action="store_true",
        default=False,
        help="Search best threshold on validation set after training and use it for final test metrics.",
    )
    p.add_argument(
        "--auto-threshold-metric",
        choices=["f1_fall", "recall_fall", "precision_fall", "cost"],
        default="f1_fall",
        help="Metric to maximize when selecting threshold on validation set.",
    )
    p.add_argument(
        "--cost-fn",
        type=float,
        default=1.0,
        help="Weight for FN in auto-threshold metric='cost' (higher => fewer missed falls).",
    )
    p.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="Weight for FP in auto-threshold metric='cost' (higher => fewer false alarms).",
    )
    p.add_argument("--threshold-min", type=float, default=0.05)
    p.add_argument("--threshold-max", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.05)
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    data = np.load(args.data_npz, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)  # (N,T,V,C)
    y_train = data["y_train"].astype(np.int32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int32)

    seq_len = int(X_train.shape[1])
    n_joints = int(X_train.shape[2])
    n_channels = int(X_train.shape[3])

    edges = MEDIAPIPE_POSE_EDGES if n_joints == 33 else []
    A = build_adjacency(n_joints, edges)
    A_norm = normalize_adjacency(A)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = build_model(n_channels=n_channels, A_norm=A_norm).to(device)

    # Weighted loss for imbalance: pos_weight = n_neg/n_pos
    y = y_train.reshape(-1)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32, device=device)
    print(f"pos_weight (for fall=1): {pos_weight.item():.3f}", flush=True)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Train/val split
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    full_ds = TensorDataset(X_train_t, y_train_t)
    val_len = int(round(0.2 * len(full_ds)))
    train_len = len(full_ds) - val_len
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True)

    def binary_metrics_from_probs(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        threshold: float,
    ) -> dict[str, float]:
        y_true = y_true.astype(np.int32).reshape(-1)
        y_pred = (y_prob >= float(threshold)).astype(np.int32).reshape(-1)

        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = (2.0 * precision * recall) / max(1e-12, (precision + recall))
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))

        return {
            "acc": float(acc),
            "precision_fall": float(precision),
            "recall_fall": float(recall),
            "f1_fall": float(f1),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }

    def collect_probs(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (y_true, y_prob, avg_loss)."""
        model.eval()
        total_loss = 0.0
        total = 0
        all_prob: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += float(loss.item()) * int(xb.shape[0])
                total += int(xb.shape[0])
                prob = torch.sigmoid(logits)
                all_prob.append(prob.detach().cpu().numpy().reshape(-1))
                all_true.append(yb.detach().cpu().numpy().reshape(-1))

        y_prob = np.concatenate(all_prob, axis=0) if all_prob else np.zeros((0,), dtype=np.float32)
        y_true = np.concatenate(all_true, axis=0) if all_true else np.zeros((0,), dtype=np.float32)
        return y_true, y_prob, total_loss / max(1, total)

    def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
        t_min = float(args.threshold_min)
        t_max = float(args.threshold_max)
        step = float(args.threshold_step)
        if step <= 0:
            raise ValueError("--threshold-step must be > 0")

        metric_name = str(args.auto_threshold_metric)

        best_t = float(args.threshold)
        best_metrics = binary_metrics_from_probs(y_true, y_prob, threshold=best_t)

        def score_of(m: dict[str, float]) -> float:
            if metric_name == "cost":
                # Lower cost is better; we return negative cost so we can keep the same "maximize" logic.
                cost = float(args.cost_fn) * float(m["fn"]) + float(args.cost_fp) * float(m["fp"])
                return -cost
            return float(m.get(metric_name, 0.0))

        best_score = score_of(best_metrics)

        t = t_min
        while t <= t_max + 1e-12:
            m = binary_metrics_from_probs(y_true, y_prob, threshold=float(t))
            score = score_of(m)
            if (score > best_score) or (score == best_score and m["recall_fall"] > best_metrics["recall_fall"]):
                best_score = score
                best_t = float(t)
                best_metrics = m
            t += step

        return best_t, best_metrics

    def evaluate(loader: DataLoader, *, threshold: float) -> tuple[float, dict[str, float]]:
        y_true, y_prob, avg_loss = collect_probs(loader)
        metrics = binary_metrics_from_probs(y_true, y_prob, threshold=float(threshold))
        return avg_loss, metrics

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    if not bool(args.eval_only):
        best_val_loss = float("inf")
        for epoch in range(1, int(args.epochs) + 1):
            print(f"Starting epoch {epoch:03d}...", flush=True)
            model.train()
            running = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * int(xb.shape[0])
                n_seen += int(xb.shape[0])

            train_loss = running / max(1, n_seen)
            val_loss, val_metrics = evaluate(val_loader, threshold=float(args.threshold))
            print(
                "Epoch {e:03d} | train_loss={tr:.4f} val_loss={vl:.4f} "
                "val_acc={acc:.4f} P_fall={p:.4f} R_fall={r:.4f} F1_fall={f1:.4f} "
                "(TN={tn:.0f} FP={fp:.0f} FN={fn:.0f} TP={tp:.0f})".format(
                    e=epoch,
                    tr=train_loss,
                    vl=val_loss,
                    acc=val_metrics["acc"],
                    p=val_metrics["precision_fall"],
                    r=val_metrics["recall_fall"],
                    f1=val_metrics["f1_fall"],
                    tn=val_metrics["tn"],
                    fp=val_metrics["fp"],
                    fn=val_metrics["fn"],
                    tp=val_metrics["tp"],
                ),
                flush=True,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        # Keep checkpoint "safe" for PyTorch>=2.6 weights_only loading:
                        # do NOT store numpy arrays or custom objects.
                        "seq_len": int(seq_len),
                        "n_joints": int(n_joints),
                        "n_channels": int(n_channels),
                    },
                    args.out_model,
                )
    elif not os.path.isfile(args.out_model):
        raise FileNotFoundError(
            f"--eval-only was set but checkpoint not found: {args.out_model}. "
            "Run training first or pass the correct --out-model path."
        )

    # Choose threshold on validation set (optional)
    selected_threshold = float(args.threshold)
    if bool(args.auto_threshold) and os.path.isfile(args.out_model):
        try:
            ckpt = torch.load(args.out_model, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(args.out_model, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        yv_true, yv_prob, _ = collect_probs(val_loader)
        selected_threshold, best_m = pick_best_threshold(yv_true, yv_prob)
        print(
            "\nAuto-threshold (val) best for {metric}: thr={thr:.2f} | "
            "P_fall={p:.4f} R_fall={r:.4f} F1_fall={f1:.4f} (TN={tn:.0f} FP={fp:.0f} FN={fn:.0f} TP={tp:.0f})".format(
                metric=str(args.auto_threshold_metric),
                thr=float(selected_threshold),
                p=best_m["precision_fall"],
                r=best_m["recall_fall"],
                f1=best_m["f1_fall"],
                tn=best_m["tn"],
                fp=best_m["fp"],
                fn=best_m["fn"],
                tp=best_m["tp"],
            ),
            flush=True,
        )

    # Final test evaluation (accuracy + confusion matrix)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        # load best checkpoint if exists
        if os.path.isfile(args.out_model):
            try:
                ckpt = torch.load(args.out_model, map_location=device, weights_only=True)
            except TypeError:
                # Older PyTorch without weights_only argument
                ckpt = torch.load(args.out_model, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        logits = model(X_test_t)
        prob = torch.sigmoid(logits)
        y_true = y_test_t.cpu().numpy().astype(np.int32).reshape(-1)
        y_prob = prob.detach().cpu().numpy().astype(np.float32).reshape(-1)
        test_metrics = binary_metrics_from_probs(y_true, y_prob, threshold=float(selected_threshold))

    print("\nTest results:", flush=True)
    print(f"Threshold: {float(selected_threshold):.2f}", flush=True)
    print(
        "Accuracy: {acc:.4f} | P_fall={p:.4f} R_fall={r:.4f} F1_fall={f1:.4f}".format(
            acc=test_metrics["acc"],
            p=test_metrics["precision_fall"],
            r=test_metrics["recall_fall"],
            f1=test_metrics["f1_fall"],
        ),
        flush=True,
    )
    print("Confusion matrix (0=normal, 1=fall)", flush=True)
    print(f"TN={int(test_metrics['tn'])}  FP={int(test_metrics['fp'])}", flush=True)
    print(f"FN={int(test_metrics['fn'])}  TP={int(test_metrics['tp'])}", flush=True)
    print(f"\nSaved best checkpoint: {args.out_model}", flush=True)


if __name__ == "__main__":
    main()
