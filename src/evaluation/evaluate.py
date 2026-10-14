import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from tqdm import tqdm

from src import config
from src.data.dataset import build_datasets
from src.model import build_model


def load_model(checkpoint_path, num_classes: int, device):
    model = build_model(num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Inference"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast("cuda", enabled=config.AMP and device.type == "cuda"):
            logits = model(pixel_values=pixel_values).logits
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred, label_names):
    acc = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(label_names))))
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    per_class = {label_names[i]: float(f1_per_class[i]) for i in range(len(label_names))}
    return {"accuracy": float(acc), "per_class_f1": per_class, "report": report}


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Test Set")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    p.add_argument(
        "-c", "--checkpoint", type=Path, default=config.CHECKPOINT_DIR / "best_model.pt",
        help="Path to the model checkpoint to evaluate (default: outputs/checkpoints/best_model.pt)",
    )
    p.add_argument(
        "-o", "--output-dir", type=Path, default=config.OUTPUT_DIR,
        help="Directory to write metrics.json and confusion_matrix.png (default: outputs/)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Run train.py first.")

    _, _, test_ds = build_datasets()
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    model = load_model(checkpoint_path, config.NUM_CLASSES, device)
    y_pred, y_true = predict(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, list(config.LABEL_NAMES))

    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}\n")
    print(metrics["report"])

    # Save metrics JSON (exclude verbose report string for cleaner JSON)
    metrics_to_save = {k: v for k, v in metrics.items() if k != "report"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Metrics saved -> {args.output_dir / 'metrics.json'}")

    plot_confusion_matrix(
        y_true, y_pred,
        list(config.LABEL_NAMES),
        args.output_dir / "confusion_matrix.png",
    )
