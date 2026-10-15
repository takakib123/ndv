"""Federated training for naturalistic-driving action classification.

Each participant is treated as a separate client that trains locally on only its own clips.
A central server aggregates client weights via FedAvg into a global model, which is evaluated on
held-out participants. The participant split reuses the centralized 70/15/15 split
(clients / central-val / test), so results are directly comparable to train.py.

Run:
    python -m src.training.federated_train
    python -m src.training.federated_train --rounds 2 --local-epochs 1 --no-wandb   # quick smoke test
    python -m src.training.federated_train --train-redaction face_blur_s0.5         # face-blurred video

Clients are trained one after another on the GPU (two VideoMAE training replicas do not fit in
8 GB), but their data is prepared in parallel: one persistent DataLoader walks the whole round's
(client, batch) schedule, so worker processes decode/augment upcoming clients' batches while the
GPU trains the current one. With ~8 clips per client a local epoch is a single batch, so per-client
loaders left the GPU idle ~75% of the time waiting on one worker.

All artifacts are written to a fresh timestamped folder under outputs/FL_results/.
"""

import argparse
import json
import logging
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from tqdm import tqdm

from src import config
from src.data.dataset import build_federated_datasets, warm_frame_cache
from src.data.redaction import parse_condition
# Reuse model / optimizer / loss / eval helpers from the centralized pipeline.
from src.training.train import build_model, build_optimizer, compute_class_weights, grad_norm, evaluate, wandb
from src.evaluation.evaluate import predict, compute_metrics, plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Federated (FedAvg) training")
    p.add_argument("--rounds", type=int, default=config.FL_ROUNDS)
    p.add_argument("--local-epochs", type=int, default=config.FL_LOCAL_EPOCHS)
    p.add_argument("--client-fraction", type=float, default=config.FL_CLIENT_FRACTION,
                   help="Fraction of clients sampled each round (1.0 = full participation)")
    p.add_argument("--train-redaction", default="none",
                   help="Redaction applied to client/val/test clips, e.g. face_blur_s0.5. 'none' = clean.")
    p.add_argument("--lr-backbone", type=float, default=config.LR_BACKBONE)
    p.add_argument("--lr-head", type=float, default=config.LR_HEAD)
    p.add_argument("--client-batch-size", type=int, default=config.FL_CLIENT_BATCH_SIZE)
    p.add_argument("--data-workers", type=int, default=config.NUM_WORKERS,
                   help="DataLoader worker processes preparing upcoming clients' batches")
    p.add_argument("--resume", default=None,
                   help="Existing run folder to continue from its checkpoints/last_model.pt. FedAvg keeps "
                        "no optimizer state across rounds, so resuming only needs the global weights.")
    p.add_argument("--wandb-id", default=None, help="W&B run id to append to when resuming")
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    return p.parse_args()


def setup_logger(log_path):
    logger = logging.getLogger("federated")
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


class RoundBatchSampler(Sampler):
    """Yields the current round's batches (lists of ConcatDataset indices) in schedule order.

    The schedule is replaced every round via `set_schedule`; the sampler runs in the main
    process, so a persistent DataLoader picks up the new schedule on its next `iter()`.
    """

    def __init__(self):
        self.schedule = []

    def set_schedule(self, schedule):
        self.schedule = schedule

    def __iter__(self):
        return (idx for _, idx in self.schedule)

    def __len__(self):
        return len(self.schedule)


def round_schedule(selected, client_offsets, client_sizes, local_epochs, batch_size, rng):
    """[(client_id, [global indices])] for one round: each selected client's local epochs,
    reshuffled per epoch, contiguous so a client's batches arrive back to back."""
    schedule = []
    for cid in selected:
        idx = list(range(client_offsets[cid], client_offsets[cid] + client_sizes[cid]))
        for _ in range(local_epochs):
            rng.shuffle(idx)
            schedule += [(cid, idx[i:i + batch_size]) for i in range(0, len(idx), batch_size)]
    return schedule


def local_train(model, batches, criterion, device, logger, client_id, lrs):
    """Train the model in place on one client's batches (all its local epochs).
    `lrs` = (lr_backbone, lr_head). Returns the mean local loss."""
    optimizer = build_optimizer(model, *lrs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP and device.type == "cuda")

    model.train()
    running_loss, n_steps = 0.0, 0
    for batch in batches:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=config.AMP and device.type == "cuda"):
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_steps += 1

    mean_loss = running_loss / max(n_steps, 1)
    logger.info(f"    client {client_id}: local_loss={mean_loss:.4f} ({n_steps} steps)")
    return mean_loss


def fedavg_aggregate(model, global_state, round_loader, schedule, client_weights, selected,
                     criterion, device, logger, lrs):
    """Run one federated round in place: train each selected client from `global_state`,
    then load the sample-count-weighted average of client weights into `model`.
    `round_loader` yields the batches of `schedule` in order (see RoundBatchSampler).
    Returns a dict {client_id: local_loss} for this round.
    """
    total = sum(client_weights[c] for c in selected)

    # Running weighted sum of client float params (CPU, fp32) — keeps only one client in memory.
    accum = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
             for k, v in global_state.items() if v.is_floating_point()}

    # Regroup the round's single batch stream into per-client runs; workers keep prefetching
    # the following clients' batches while the current client trains.
    batch_iter = iter(round_loader)
    per_client = {}
    for cid, _ in schedule:
        per_client[cid] = per_client.get(cid, 0) + 1

    def client_batches(n):
        for _ in range(n):
            yield next(batch_iter)

    client_losses = {}
    for client_id in selected:
        model.load_state_dict(global_state)
        loss = local_train(model, client_batches(per_client[client_id]), criterion, device,
                           logger, client_id, lrs)
        client_losses[client_id] = loss

        frac = client_weights[client_id] / total
        sd = model.state_dict()
        for k in accum:
            accum[k].add_(sd[k].detach().float().cpu(), alpha=frac)

    # Cast back to original dtypes; non-float buffers (none in VideoMAE) keep the global value.
    new_state = {}
    for k, v in global_state.items():
        new_state[k] = accum[k].to(v.dtype) if v.is_floating_point() else v.clone()
    model.load_state_dict(new_state)

    return client_losses


def log_client_losses(rnd, client_losses, use_wandb):
    """Push per-client loss monitoring figures to W&B for the round."""
    if not use_wandb:
        return
    losses = list(client_losses.values())
    log = {f"client_loss/{cid}": loss for cid, loss in client_losses.items()}
    # Distribution of local losses across clients this round + spread statistics.
    log["client_loss/histogram"] = wandb.Histogram(losses)
    log["client_loss/mean"] = float(np.mean(losses))
    log["client_loss/min"] = float(np.min(losses))
    log["client_loss/max"] = float(np.max(losses))
    log["client_loss/std"] = float(np.std(losses))
    # Per-round bar chart of each client's loss (a fresh figure each round).
    table = wandb.Table(data=[[cid, loss] for cid, loss in sorted(client_losses.items())],
                        columns=["client", "loss"])
    log[f"client_loss/bar_round_{rnd}"] = wandb.plot.bar(
        table, "client", "loss", title=f"Client local loss — round {rnd}")
    wandb.log(log, step=rnd)


def main():
    args = parse_args()

    # Reproducibility (matches train.py)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestamped run folder (one separate folder per run/model)
    redactor = parse_condition(args.train_redaction)
    redacted = not redactor.is_identity

    if args.resume:
        run_dir = Path(args.resume)
        run_name = run_dir.name
    else:
        run_name = datetime.now().strftime("%d_%B_%H%M%S")
        if redacted:
            run_name += f"_{redactor.key()}"
        run_dir = config.FED_OUTPUT_DIR / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "train.log")
    logger.info(f"Federated run: {run_name}")
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Train redaction: {redactor.key()}")

    # Data: per-client training datasets + centralized val/test
    client_datasets, val_ds, test_ds, label2idx = build_federated_datasets(
        redactor=redactor if redacted else None)
    client_ids = sorted(client_datasets.keys())
    logger.info(f"Clients: {len(client_ids)} | val clips: {len(val_ds)} | test clips: {len(test_ds)}")

    # All clients behind one persistent DataLoader; each round sets its (client, batch) schedule.
    client_offsets, client_sizes, off = {}, {}, 0
    for cid in client_ids:
        client_offsets[cid], client_sizes[cid] = off, len(client_datasets[cid])
        off += client_sizes[cid]
    round_sampler = RoundBatchSampler()
    round_loader = DataLoader(
        ConcatDataset([client_datasets[c] for c in client_ids]), batch_sampler=round_sampler,
        num_workers=args.data_workers, pin_memory=True,
        persistent_workers=args.data_workers > 0,
        prefetch_factor=4 if args.data_workers > 0 else None,
    )
    client_weights = dict(client_sizes)

    # Pre-decode every clip once (parallel) so rounds are GPU-bound, not video-decode bound.
    all_records = ([r for ds in client_datasets.values() for r in ds.records]
                   + list(val_ds.records) + list(test_ds.records))
    warm_frame_cache(all_records, workers=config.CACHE_WARM_WORKERS)

    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=config.NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=config.NUM_WORKERS > 0,
    )

    # Model + global loss (class weights from pooled client label distribution — histograms only)
    model = build_model(config.NUM_CLASSES).to(device)
    pooled = SimpleNamespace(records=[r for ds in client_datasets.values() for r in ds.records])
    class_weights = compute_class_weights(pooled).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    n_per_round = max(1, round(args.client_fraction * len(client_ids)))
    fl_cfg = {
        "algorithm": "FedAvg",
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "client_fraction": args.client_fraction,
        "clients_per_round": n_per_round,
        "num_clients": len(client_ids),
        "client_batch_size": args.client_batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": config.WEIGHT_DECAY,
        "grad_clip": config.GRAD_CLIP,
        "amp": config.AMP,
        "lr_schedule": "constant (fresh AdamW per client each round; no cosine warmup)",
        "model": config.MODEL_NAME,
        "num_classes": config.NUM_CLASSES,
        "num_frames": config.NUM_FRAMES,
        "image_size": config.IMAGE_SIZE,
        "video_type": config.DATA_VIDEO_TYPE,
        "camera": config.DATA_CAMERA,
        "seed": config.SEED,
        "train_redaction": redactor.key(),
        "data_workers": args.data_workers,
        "val_clips": len(val_ds),
        "test_clips": len(test_ds),
    }
    if not args.resume:   # keep the original run's config file untouched on resume
        with open(run_dir / "fl_config.json", "w") as f:
            json.dump(fl_cfg, f, indent=2)
    logger.info(f"Config: {json.dumps(fl_cfg)}")

    # Federated rounds
    best_val_acc = 0.0
    start_round = 1
    if args.resume:
        last = torch.load(ckpt_dir / "last_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(last["model_state_dict"])
        start_round = last["epoch"] + 1
        if (ckpt_dir / "best_model.pt").exists():
            best_val_acc = torch.load(ckpt_dir / "best_model.pt", map_location="cpu",
                                      weights_only=False)["val_acc"]
        logger.info(f"Resumed from round {last['epoch']} (best val_acc so far {best_val_acc:.4f})")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project="ndv-action-federated", name=run_name, config=fl_cfg,
                   id=args.wandb_id, resume="allow" if args.wandb_id else None)

    history = []
    sample_rng = random.Random(config.SEED)

    logger.info(f"Starting federated training: {args.rounds} rounds x {args.local_epochs} local "
                f"epochs x {n_per_round}/{len(client_ids)} clients/round")

    for rnd in tqdm(range(start_round, args.rounds + 1), desc="FL rounds"):
        if n_per_round >= len(client_ids):
            selected = client_ids
        else:
            selected = sample_rng.sample(client_ids, n_per_round)

        schedule = round_schedule(selected, client_offsets, client_sizes, args.local_epochs,
                                  args.client_batch_size, sample_rng)
        round_sampler.set_schedule(schedule)

        round_t0 = time.perf_counter()
        global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        client_losses = fedavg_aggregate(
            model, global_state, round_loader, schedule, client_weights, selected,
            criterion, device, logger, (args.lr_backbone, args.lr_head),
        )
        round_time = time.perf_counter() - round_t0
        train_loss = float(np.mean(list(client_losses.values())))
        log_client_losses(rnd, client_losses, use_wandb)

        val_loss, val_acc, per_class_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        logger.info(f"Round {rnd}/{args.rounds} | train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | train_time={round_time:.0f}s")

        history.append({"round": rnd, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "client_losses": client_losses})

        if use_wandb:
            log = {"round": rnd, "train/loss": train_loss, "val/loss": val_loss,
                   "val/accuracy": val_acc, "val/best_accuracy": max(best_val_acc, val_acc),
                   "sys/round_train_seconds": round_time, **per_class_acc}
            if device.type == "cuda":
                log["sys/gpu_mem_gb"] = torch.cuda.memory_reserved(device) / 1e9
            wandb.log(log, step=rnd)

        ckpt = {"epoch": rnd, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {}, "val_acc": val_acc}
        torch.save(ckpt, ckpt_dir / "last_model.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, ckpt_dir / "best_model.pt")
            logger.info(f"  ** New best val_acc={best_val_acc:.4f} — checkpoint saved")

    # On resume this holds only the resumed rounds; train.log has the full history.
    with open(run_dir / ("round_history.json" if start_round == 1 else f"round_history_from_{start_round}.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")

    # Final test evaluation with the best global model
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (round {best_ckpt['epoch']}, "
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

    if use_wandb:
        wandb.log({"test/accuracy": metrics["accuracy"]})
        wandb.finish()

    logger.info(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
