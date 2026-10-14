import argparse
import random
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import wandb

from src import config
from src.model import build_model
from src.data.dataset import build_datasets
from src.data.redaction import parse_condition
from src.evaluation.evaluate import predict, compute_metrics, plot_confusion_matrix

import os


def _load_env(env_path):
    """Minimal .env loader (no external deps). Sets KEY=VALUE pairs into this
    process's environment only — nothing is exported to the shared shell — and
    never overrides a value already present in the environment.
    """
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# Loads WANDB_API_KEY (and anything else) from <repo>/.env BEFORE wandb is used.
_load_env(config.BASE_DIR / ".env")
config.WANDB_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_DIR", str(config.WANDB_DIR))   # keep wandb runs under artifacts/


try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

    class _WandbStub:
        """No-op wandb stub so the script runs without a wandb installation."""
        def init(self, **kwargs):   return self
        def log(self, *a, **kw):    pass
        def finish(self, **kwargs): pass
        def __enter__(self):        return self
        def __exit__(self, *a):     pass

    wandb = _WandbStub()


def setup_logger(log_path):
    """Logger that writes timestamped, leveled messages to both a file and the console."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def build_optimizer(model: nn.Module, lr_backbone: float = None, lr_head: float = None) -> torch.optim.AdamW:
    backbone_params = [p for name, p in model.named_parameters() if "classifier" not in name]
    head_params = list(model.classifier.parameters())
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone or config.LR_BACKBONE},
            {"params": head_params, "lr": lr_head or config.LR_HEAD},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def compute_class_weights(dataset) -> torch.FloatTensor:
    labels = [rec["label_idx"] for rec in dataset.records]
    weights = compute_class_weight(
        "balanced",
        classes=np.arange(config.NUM_CLASSES),
        y=np.array(labels),
    )
    return torch.FloatTensor(weights)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, global_step: int):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=config.AMP and device.type == "cuda"):
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step_log = {
            "train/step_loss": loss.item(),
            "train/grad_norm": grad_norm(model),
            "train/lr_backbone": optimizer.param_groups[0]["lr"],
            "train/lr_head": optimizer.param_groups[1]["lr"],
            "train/global_step": global_step,
        }
        if device.type == "cuda":
            step_log["sys/gpu_mem_gb"] = torch.cuda.memory_reserved(device) / 1e9
        wandb.log(step_log, step=global_step)

        total_loss += loss.item()
        global_step += 1

    return total_loss / len(loader), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for batch in tqdm(loader, desc="  val  ", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=config.AMP and device.type == "cuda"):
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(outputs.logits, dim=1)
        preds = probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    per_class_acc = {}
    for i, name in enumerate(config.LABEL_NAMES):
        mask = all_labels == i
        per_class_acc[f"val/acc_{name}"] = (all_preds[mask] == all_labels[mask]).float().mean().item() if mask.sum() > 0 else 0.0

    return total_loss / len(loader), correct / total, per_class_acc, all_preds, all_labels, all_probs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-redaction", default="none",
                   help="Redaction applied to train/val/test clips, e.g. face_blur_s0.5, face_black. "
                        "'none' trains on clean video.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    redactor = parse_condition(args.train_redaction)
    redacted = not redactor.is_identity

    # Reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestamped run folder (one self-contained folder per run/model)
    run_name = datetime.now().strftime("%d_%B_%H%M%S")
    if redacted:
        run_name += f"_{redactor.key()}"
    run_dir = config.OUTPUT_DIR / "results" / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "train.log")
    logger.info(f"Run: {run_name}")
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Train redaction: {redactor.key()}")

    # Datasets and loaders
    train_ds, val_ds, test_ds = build_datasets(redactor=redactor if redacted else None)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    # Model, optimizer, scheduler, loss
    model = build_model(config.NUM_CLASSES).to(device)
    optimizer = build_optimizer(model)

    total_steps = config.NUM_EPOCHS * len(train_loader)
    warmup_steps = config.WARMUP_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP and device.type == "cuda")

    # Full config — saved locally and pushed to W&B so every run is reproducible
    cfg = {
        "model": config.MODEL_NAME,
        "labels": list(config.LABEL_NAMES),
        "num_classes": config.NUM_CLASSES,
        "num_frames": config.NUM_FRAMES,
        "image_size": config.IMAGE_SIZE,
        "batch_size": config.BATCH_SIZE,
        "num_epochs": config.NUM_EPOCHS,
        "lr_backbone": config.LR_BACKBONE,
        "lr_head": config.LR_HEAD,
        "weight_decay": config.WEIGHT_DECAY,
        "warmup_epochs": config.WARMUP_EPOCHS,
        "grad_clip": config.GRAD_CLIP,
        "amp": config.AMP,
        "seed": config.SEED,
        "train_clips": len(train_ds),
        "val_clips": len(val_ds),
        "test_clips": len(test_ds),
        "train_redaction": redactor.key(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Config: {json.dumps(cfg)}")

    wandb.init(project="ndv-action-classification", name=run_name, config=cfg)

    best_val_acc = 0.0
    global_step = 0
    history = []

    logger.info(f"Training for {config.NUM_EPOCHS} epochs — "
                f"{len(train_ds)} train / {len(val_ds)} val clips")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{config.NUM_EPOCHS}")

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, device, global_step
        )
        val_loss, val_acc, per_class_acc, all_preds, all_labels, all_probs = evaluate(
            model, val_loader, criterion, device
        )

        logger.info(f"Epoch {epoch}/{config.NUM_EPOCHS} | train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        epoch_log = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/best_accuracy": max(best_val_acc, val_acc),
            **per_class_acc,
        }
        if device.type == "cuda":
            epoch_log["sys/gpu_mem_gb"] = torch.cuda.memory_reserved(device) / 1e9

        # Confusion matrix
        epoch_log["val/confusion_matrix"] = wandb.plot.confusion_matrix(
            preds=all_preds.tolist(),
            y_true=all_labels.tolist(),
            class_names=list(config.LABEL_NAMES),
        )

        wandb.log(epoch_log, step=global_step)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "val_acc": val_acc}

        # Always save last checkpoint (per-run folder)
        torch.save(ckpt, ckpt_dir / "last_model.pt")

        # Save best checkpoint (local only)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(ckpt, best_ckpt_path)
            # Stable copy for downstream scripts (evaluate.py / attack_inference.py). Only for
            # clean runs, so a redacted model never silently replaces the baseline.
            if not redacted:
                torch.save(ckpt, config.CHECKPOINT_DIR / "best_model.pt")
            logger.info(f"  ** New best val_acc={best_val_acc:.4f} — checkpoint saved")

    # Save training history
    with open(run_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")

    # Final test evaluation with the best model
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (epoch {best_ckpt['epoch']}, "
                    f"val_acc={best_ckpt['val_acc']:.4f}) for testing")
    model.eval()

    y_pred, y_true = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, list(config.LABEL_NAMES))
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info("\n" + metrics["report"])

    metrics_to_save = {k: v for k, v in metrics.items() if k != "report"}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    plot_confusion_matrix(y_true, y_pred, list(config.LABEL_NAMES), run_dir / "confusion_matrix.png")

    wandb.log({"test/accuracy": metrics["accuracy"]})
    wandb.finish()
    logger.info(f"All results saved to {run_dir}")
