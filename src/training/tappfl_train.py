"""TAPPFL: task-agnostic privacy-preserving federated learning for naturalistic driving.

Adversarially adapts the VideoMAE backbone (Feature Extractor f_Theta) so its 768-d
representation z = f_Theta(x) conceals a sensitive attribute u (participant gender, K=2)
while retaining task-relevant information about the input x. Only the backbone is
communicated / FedAvg'd; the action head (decoder) stays frozen and is used only for
offline utility evaluation.

Three networks:
  f_Theta  VideoMAE backbone           FEDERATED (FedAvg of backbone weights)
  g_Psi    privacy adversary (MLP)     federation-SHARED, never communicated
  h_Omega  JSD MI estimator (critic)   federation-SHARED, never communicated

Because each client is one participant and every participant has a single gender, a
strictly per-client adversary would only ever see one class and could not learn. We
therefore keep ONE shared g_Psi / h_Omega, persisted across rounds and updated
sequentially as clients are visited; across clients they see both genders, so even a
single-gender client batch yields a meaningful adversarial FE gradient. The FE remains
the only FedAvg'd component, so subject-per-client realism is preserved.

Run:
    python -m src.training.tappfl_train                                   # full run (30 rounds)
    python -m src.training.tappfl_train --rounds 2 --no-wandb             # quick smoke test
    python -m src.training.tappfl_train --lambda 0.0                      # privacy-only sanity check

Artifacts are written to a fresh timestamped folder under outputs/TAPPFL_results/.
"""

import argparse
import json
import logging
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.data.dataset import build_tappfl_datasets, warm_frame_cache
# Reuse model construction and offline-eval helpers from the existing pipeline.
from src.training.train import build_model, wandb
from src.evaluation.evaluate import predict, compute_metrics, plot_confusion_matrix


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class PrivacyNet(nn.Module):
    """g_Psi: predicts the sensitive attribute u from the representation z."""

    def __init__(self, d_z: int = 768, k: int = 2, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, k),
        )

    def forward(self, z):
        return self.net(z)


class MIEstimator(nn.Module):
    """h_Omega: JSD critic T(x, z, u) -> scalar, following the original TAPPFL MI estimator.
    The private label u is fed in as an extra input (concatenated with the encoded x and z
    streams), matching `mi_model` in github.com/TAPPFL/TAPPFL."""

    def __init__(self, d_x: int, d_z: int = 768, d_hidden: int = 256):
        super().__init__()
        self.x_enc = nn.Linear(d_x, d_hidden)
        self.z_enc = nn.Linear(d_z, d_hidden)
        self.joint = nn.Sequential(
            nn.Linear(2 * d_hidden + 1, d_hidden), nn.ReLU(),   # +1 for the private label u
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x, z, u):
        hx = F.relu(self.x_enc(x))
        hz = F.relu(self.z_enc(z))
        u = u.float().unsqueeze(1)                              # (B, 1)
        return self.joint(torch.cat([hx, hz, u], dim=1)).squeeze(1)   # (B,)


class FeatureBuffer:
    """Cross-client, per-class FIFO buffer of detached representations (z) with their gender
    label. Sampling a class-balanced batch lets the SHARED adversary g_Psi keep seeing both
    genders every step, so it stops forgetting across single-gender clients."""

    def __init__(self, k: int, cap: int):
        self.k = k
        self.pools = [deque(maxlen=cap) for _ in range(k)]

    def add(self, z, u):
        for zi, ui in zip(z.detach(), u.tolist()):
            self.pools[ui].append(zi)

    def ready(self):
        return all(len(p) > 0 for p in self.pools)

    def sample_balanced(self, n_per_class):
        zs, us = [], []
        for c in range(self.k):
            pool = self.pools[c]
            if not pool:
                continue
            for _ in range(n_per_class):
                zs.append(pool[random.randrange(len(pool))])
                us.append(c)
        z = torch.stack(zs)
        u = torch.tensor(us, device=z.device)
        return z, u


class TaskHead(nn.Module):
    """Fresh MLP head (768->256->num_classes) trained on the FROZEN encoder to measure how
    much task-relevant information the representation retains (the utility probe)."""

    def __init__(self, d_z: int = 768, num_classes: int = 8, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# FE encoding / MI helpers
# ---------------------------------------------------------------------------

def _amp_enabled(device):
    return config.AMP and device.type == "cuda"


def encode(model, x):
    """z = f_Theta(x): mean-pooled VideoMAE backbone features (B, 768). Gradients flow."""
    out = model.videomae(pixel_values=x)
    return out.last_hidden_state.mean(dim=1)


def make_x_flat(x, pool: int):
    """Pooled-pixel surrogate of the video: temporal mean + spatial avg-pool -> flatten.
    x: (B, T, C, H, W) -> (B, C*pool*pool). Computed without gradient (data side)."""
    with torch.no_grad():
        frame = x.mean(dim=1)                                  # (B, C, H, W)
        pooled = F.adaptive_avg_pool2d(frame, (pool, pool))    # (B, C, pool, pool)
        return pooled.flatten(1).float()                       # (B, C*pool*pool)


def jsd_mi(h, x_flat, z, u, x_tilde):
    """Jensen-Shannon lower bound on I(x; z | u), matching the original TAPPFL jsd_MI:
    Ej - Em with softplus, u passed into the critic. Returns a value <= 0."""
    t_pos = h(x_flat, z, u)               # joint    p(x, z)
    t_neg = h(x_tilde, z, u)              # marginal p(x) p(z), x_tilde = roll(x_flat, 1)
    return torch.mean(-F.softplus(-t_pos)) - torch.mean(F.softplus(t_neg))


# ---------------------------------------------------------------------------
# Adversary relaunch (defeat the "fooling one fixed adversary" trap)
# ---------------------------------------------------------------------------

def reinit_adversary(device):
    """Build a fresh privacy attacker g_Psi and its optimizer (random init)."""
    g = PrivacyNet(d_z=768, k=config.TAPPFL_K).to(device)
    opt_g = torch.optim.AdamW(g.parameters(), lr=config.TAPPFL_ADV_LR, weight_decay=config.WEIGHT_DECAY)
    return g, opt_g


def warmup_adversary(g, opt_g, buffer, criterion, steps):
    """Train a freshly relaunched g_Psi to competence on class-balanced batches from the
    buffer (the current representation) so it is a strong attacker before the FE faces it.
    Returns (first_loss, last_loss) for monitoring; no-op if the buffer lacks both classes."""
    if not buffer.ready():
        return None, None
    g.train()
    first = last = None
    for s in range(steps):
        z, u = buffer.sample_balanced(config.FL_CLIENT_BATCH_SIZE)
        opt_g.zero_grad()
        loss = criterion(g(z), u)
        loss.backward()
        nn.utils.clip_grad_norm_(g.parameters(), config.TAPPFL_CLIP)
        opt_g.step()
        last = float(loss.item())
        if s == 0:
            first = last
    return first, last


# ---------------------------------------------------------------------------
# Per-client TAPPFL training (shared g / h updated in place)
# ---------------------------------------------------------------------------

def client_train(model, loader, g, h, opt_g, opt_h, lam, device, criterion, buffer,
                 logger, client_id):
    """One client's local TAPPFL update, following the original per-batch minimax
    (github.com/TAPPFL/TAPPFL): for each batch, one feature-extractor update on
    total_loss = -lambda*L_adv + (1-lambda)*L_JSD, one adversary update (CE on the true
    private label), and one MI-estimator update -- no warm-up, no repeated adversary steps.

    The FE was reloaded from Theta_global by the caller and is trained in place; the SHARED
    g / h are updated in place. The adversary trains on a class-balanced sample from the
    cross-client `buffer` (with the current client's z appended) so it keeps seeing both
    genders and stops forgetting across single-gender clients. Returns per-client diagnostics."""
    pool = config.TAPPFL_X_POOL
    amp = _amp_enabled(device)
    fe_params = [p for p in model.parameters() if p.requires_grad]
    fe_opt = torch.optim.AdamW(fe_params, lr=config.TAPPFL_LR_INIT, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    model.train()
    g.train()
    h.train()
    acc = {"l_total": 0.0, "l_priv": 0.0, "l_jsd": 0.0, "jsd_mi": 0.0,
           "fe_norm": 0.0, "g_norm": 0.0, "h_norm": 0.0,
           "adv_correct": 0, "adv_seen": 0, "n": 0}

    for _ in range(config.TAPPFL_LOCAL_EPOCHS):
        for batch in loader:
            x = batch["pixel_values"].to(device)
            u = batch["gender"].to(device)
            x_flat = make_x_flat(x, pool)
            x_tilde = torch.roll(x_flat, shifts=1, dims=0)

            # ── Feature-extractor update: total_loss = -lambda*L_adv + (1-lambda)*L_JSD ──
            # Gradients flow through g and h into the backbone; only fe_opt steps here.
            fe_opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                z = encode(model, x)
                logits_adv = g(z)
                l_adv = criterion(logits_adv, u)
                l_jsd = -jsd_mi(h, x_flat, z, u, x_tilde)              # >= 0
                l_total = -lam * l_adv + (1.0 - lam) * l_jsd
            scaler.scale(l_total).backward()
            scaler.unscale_(fe_opt)
            fe_norm = nn.utils.clip_grad_norm_(fe_params, config.TAPPFL_CLIP)
            scaler.step(fe_opt)
            scaler.update()

            z_det = z.detach().float()                                # stop-grad into the FE
            acc["adv_correct"] += int((logits_adv.detach().argmax(1) == u).sum().item())
            acc["adv_seen"] += int(u.size(0))

            # ── Adversary update: CE on the true private label, over a class-balanced batch
            #    from the cross-client buffer (FE detached) so g_Psi does not forget. ──
            buffer.add(z_det, u)
            if buffer.ready():
                z_g, u_g = buffer.sample_balanced(config.FL_CLIENT_BATCH_SIZE)
            else:
                z_g, u_g = z_det, u                                    # cold start: current client only
            opt_g.zero_grad()
            l_adv2 = criterion(g(z_g), u_g)
            l_adv2.backward()
            g_norm = nn.utils.clip_grad_norm_(g.parameters(), config.TAPPFL_CLIP)
            opt_g.step()

            # ── MI-estimator update: maximize the JSD MI bound (FE detached) ──
            opt_h.zero_grad()
            l_mi = -jsd_mi(h, x_flat, z_det, u, x_tilde)
            l_mi.backward()
            h_norm = nn.utils.clip_grad_norm_(h.parameters(), config.TAPPFL_CLIP)
            opt_h.step()

            acc["l_total"] += float(l_total.item())
            acc["l_priv"] += float(l_adv.item())
            acc["l_jsd"] += float(l_jsd.item())
            acc["jsd_mi"] += float((-l_jsd).item())                    # report jsd_MI (<= 0)
            acc["fe_norm"] += float(fe_norm)
            acc["g_norm"] += float(g_norm)
            acc["h_norm"] += float(h_norm)
            acc["n"] += 1

    n = max(acc["n"], 1)
    stats = {
        "L_total": acc["l_total"] / n,
        "L_priv": acc["l_priv"] / n,
        "L_JSD": acc["l_jsd"] / n,
        "jsd_MI": acc["jsd_mi"] / n,
        "fe_grad_norm": acc["fe_norm"] / n,
        "g_grad_norm": acc["g_norm"] / n,
        "h_grad_norm": acc["h_norm"] / n,
        "adv_train_acc": acc["adv_correct"] / max(acc["adv_seen"], 1),
        "n_steps": acc["n"],
    }
    logger.info(f"    client {client_id}: L_total={stats['L_total']:.4f} "
                f"L_priv={stats['L_priv']:.4f} jsd_MI={stats['jsd_MI']:.4f} "
                f"adv_acc={stats['adv_train_acc']:.3f} ({acc['n']} steps)")
    return stats


# ---------------------------------------------------------------------------
# Federated aggregation (backbone weights only; classifier frozen/identical)
# ---------------------------------------------------------------------------

def fedavg_aggregate(model, global_state, client_loaders, selected,
                     g, h, opt_g, opt_h, lam, device, criterion, buffer, logger):
    """One federated round: train each selected client from `global_state` (updating the
    shared g / h in place), then load the UNIFORM average of client FE weights into `model`
    (matching the original TAPPFL `average_weights`). Returns {client_id: stats}."""
    accum = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
             for k, v in global_state.items() if v.is_floating_point()}
    frac = 1.0 / len(selected)

    client_stats = {}
    for client_id in selected:
        model.load_state_dict(global_state)
        stats = client_train(model, client_loaders[client_id], g, h, opt_g, opt_h, lam,
                             device, criterion, buffer, logger, client_id)
        client_stats[client_id] = stats

        sd = model.state_dict()
        for k in accum:
            accum[k].add_(sd[k].detach().float().cpu(), alpha=frac)

    new_state = {}
    for k, v in global_state.items():
        new_state[k] = accum[k].to(v.dtype) if v.is_floating_point() else v.clone()
    model.load_state_dict(new_state)
    return client_stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_attr_acc(model, g, loader, device):
    """Gender-classification of the shared adversary on the global FE over held-out subjects.
    Lower (toward 1/K) is better -> used for checkpoint selection. Returns a metrics dict."""
    amp = _amp_enabled(device)
    model.eval()
    g.eval()
    preds, labels = [], []
    for batch in loader:
        x = batch["pixel_values"].to(device)
        u = batch["gender"].to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            z = encode(model, x).float()
        preds.append(g(z).argmax(dim=1).cpu())
        labels.append(u.cpu())
    g.train()
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return {
        "attr_acc": float(accuracy_score(labels, preds)),
        "attr_balanced_acc": float(balanced_accuracy_score(labels, preds)),
    }


@torch.no_grad()
def extract_features(model, dataset, device):
    """Frozen-encoder representations z + action labels y + gender labels u + subject ids for
    one dataset. Iterates with shuffle=False so subjects align with `dataset.records`."""
    amp = _amp_enabled(device)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    model.eval()
    Z, Y, U = [], [], []
    for batch in loader:
        x = batch["pixel_values"].to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            z = encode(model, x).float()
        Z.append(z.cpu())
        Y.append(batch["labels"])
        U.append(batch["gender"])
    subjects = np.array([r["subject"] for r in dataset.records])
    return (torch.cat(Z).numpy(), torch.cat(Y).numpy(), torch.cat(U).numpy(), subjects)


def probe_gender_seen(Z, U, subjects, seed):
    """Attribute-leakage probe on the PARTICIPATING (training) clients: train a logistic
    probe on the frozen z of some training subjects, test on the OTHER training subjects
    (grouped by subject so it measures gender decodability, not identity memorization).
    probe_acc ~ majority baseline means the encoder concealed gender (ideal privacy)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, te = next(gss.split(Z, U, groups=subjects))
    note = None
    if len(np.unique(U[te])) < 2:
        note = "probe-test subjects are single-gender; accuracy is not informative."
    scaler = StandardScaler().fit(Z[tr])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(Z[tr]), U[tr])
    pred = clf.predict(scaler.transform(Z[te]))
    return {
        "probe_accuracy": float(accuracy_score(U[te], pred)),
        "probe_balanced_accuracy": float(balanced_accuracy_score(U[te], pred)),
        "majority_baseline": float(max(U.mean(), 1 - U.mean())),
        "chance": 1.0 / config.TAPPFL_K,
        "n_probe_train_subjects": int(len(np.unique(subjects[tr]))),
        "n_probe_test_subjects": int(len(np.unique(subjects[te]))),
        "measured_on": "training (seen) clients, subject-grouped",
        "note": note,
    }


def utility_retrain_head(Z_tr, y_tr, Z_te, y_te, device, seed):
    """Utility = task info retained in z: freeze the encoder, train a FRESH MLP head on the
    training subjects' z (standardized), report action accuracy on the held-out test z."""
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(Z_tr)
    Xtr = torch.tensor(scaler.transform(Z_tr), dtype=torch.float32, device=device)
    Xte = torch.tensor(scaler.transform(Z_te), dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    head = TaskHead(d_z=Z_tr.shape[1], num_classes=config.NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    head.train()
    for _ in range(config.TAPPFL_UTIL_HEAD_EPOCHS):
        opt.zero_grad()
        loss = crit(head(Xtr), ytr)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(Xte).argmax(1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "head": "retrained MLP 768->256->K on frozen z",
        "test_pred": pred.tolist(),
        "test_true": y_te.tolist(),
    }


def run_eval(model, train_ds, test_ds, device, seed):
    """Offline eval, sharing one frozen-encoder feature extraction:
      - privacy: gender leakage on the TRAINING (seen) clients (subject-grouped probe),
      - utility: action accuracy of a FRESH MLP head trained on frozen z (train subjects)
                 and evaluated on the held-out test clips.
    Returns (privacy_dict, utility_dict)."""
    Z_tr, y_tr, u_tr, subj_tr = extract_features(model, train_ds, device)
    Z_te, y_te, u_te, subj_te = extract_features(model, test_ds, device)
    privacy = probe_gender_seen(Z_tr, u_tr, subj_tr, seed)
    utility = utility_retrain_head(Z_tr, y_tr, Z_te, y_te, device, seed)
    return privacy, utility


# ---------------------------------------------------------------------------
# wandb diagnostics
# ---------------------------------------------------------------------------

def log_round_diagnostics(rnd, client_stats, val_metrics, best_val_attr_acc, lam, device):
    """Push per-round TAPPFL training dynamics to W&B: loss components, adversarial-game
    signals, gradient norms, per-client spread, and the validation privacy proxy."""
    means = lambda k: float(np.mean([s[k] for s in client_stats.values()]))
    chance = 1.0 / config.TAPPFL_K
    log = {
        "round": rnd,
        "train/lambda": lam,
        # ── loss components (L_total = -lambda*L_priv + (1-lambda)*L_JSD) ──
        "train/L_total": means("L_total"),
        "train/L_priv": means("L_priv"),                       # adversary CE seen by FE (higher = more confused)
        "train/L_JSD": means("L_JSD"),                         # utility loss (>=0; lower = more retained MI)
        "train/jsd_MI": means("jsd_MI"),                       # MI estimate (<=0; toward 0 as h converges)
        "train/adv_term": -lam * means("L_priv"),              # privacy contribution to L_total
        "train/util_term": (1.0 - lam) * means("L_JSD"),       # utility contribution to L_total
        # ── adversarial game ──
        "train/adv_train_acc": means("adv_train_acc"),         # g_Psi accuracy on client clips (the attacker)
        # ── gradient norms (pre-clip; clip = TAPPFL_CLIP) ──
        "train/fe_grad_norm": means("fe_grad_norm"),
        "train/g_grad_norm": means("g_grad_norm"),
        "train/h_grad_norm": means("h_grad_norm"),
        # ── validation privacy proxy (lower toward chance = more private) ──
        "val/attr_acc": val_metrics["attr_acc"],
        "val/attr_balanced_acc": val_metrics["attr_balanced_acc"],
        "val/best_attr_acc": min(best_val_attr_acc, val_metrics["attr_acc"]),
        "val/attr_acc_minus_chance": val_metrics["attr_acc"] - chance,
    }
    # Per-client spread on the most informative signals (plain scalars -> reliable line
    # panels; wandb.Histogram time-series don't support a custom round x-axis).
    for k in ("L_total", "jsd_MI", "adv_train_acc"):
        vals = np.array([s[k] for s in client_stats.values()], dtype=float)
        log[f"train/{k}_min"] = float(vals.min())
        log[f"train/{k}_p50"] = float(np.median(vals))
        log[f"train/{k}_max"] = float(vals.max())
        log[f"train/{k}_std"] = float(vals.std())
    if device.type == "cuda":
        log["sys/gpu_mem_gb"] = torch.cuda.memory_reserved(device) / 1e9
    wandb.log(log)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TAPPFL privacy-preserving federated training")
    p.add_argument("--rounds", type=int, default=config.TAPPFL_ROUNDS)
    p.add_argument("--local-epochs", type=int, default=config.TAPPFL_LOCAL_EPOCHS,
                   help="Local epochs per client per round (original TAPPFL uses 10)")
    p.add_argument("--lambda", dest="lam", type=float, default=config.TRADEOFF_LAMBDA,
                   help="Weight on the privacy term (1=pure privacy, 0=pure utility)")
    p.add_argument("--client-fraction", type=float, default=1.0,
                   help="Fraction of clients sampled each round (1.0 = full participation)")
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="FE/decoder init (.pt). Default: latest federated best_model.pt")
    p.add_argument("--adv-reset-every", type=int, default=config.TAPPFL_ADV_RESET_EVERY,
                   help="Relaunch (re-init + re-warm) the adversary g_Psi every N rounds (0 = never)")
    p.add_argument("--adv-warmup-steps", type=int, default=config.TAPPFL_ADV_WARMUP_STEPS,
                   help="Steps to train a freshly relaunched g_Psi on the buffer")
    p.add_argument("--mi-pretrain-epochs", type=int, default=0,
                   help="Centralized warm-up epochs for h_Omega before round 1 (0 = skip)")
    p.add_argument("--eval-every", type=int, default=5,
                   help="Run + log the held-out privacy probe & utility every N rounds "
                        "(0 = only at the end). Gives the privacy/utility trade-off curve.")
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    return p.parse_args()


def setup_logger(log_path):
    logger = logging.getLogger("tappfl")
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


def find_init_checkpoint(args):
    if args.init_checkpoint is not None:
        return args.init_checkpoint.resolve()
    cands = list(config.FED_OUTPUT_DIR.glob("*/checkpoints/best_model.pt"))
    if cands:
        return max(cands, key=lambda p: p.stat().st_mtime).resolve()
    fallback = config.CHECKPOINT_DIR / "best_model.pt"
    return fallback.resolve()


def freeze_decoder(model):
    """Freeze the action head (decoder); keep the backbone (FE) trainable."""
    for p in model.classifier.parameters():
        p.requires_grad_(False)


def mi_pretrain(model, client_loaders, h, opt_h, epochs, device, logger):
    """Optional centralized warm start for h_Omega with f_Theta frozen (spec section 6)."""
    pool = config.TAPPFL_X_POOL
    amp = _amp_enabled(device)
    model.eval()
    h.train()
    loaders = list(client_loaders.values())
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for loader in loaders:
            for batch in loader:
                x = batch["pixel_values"].to(device)
                u = batch["gender"].to(device)
                x_flat = make_x_flat(x, pool)
                x_tilde = torch.roll(x_flat, shifts=1, dims=0)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                    z = encode(model, x).float()
                opt_h.zero_grad()
                loss = -jsd_mi(h, x_flat, z, u, x_tilde)
                loss.backward()
                nn.utils.clip_grad_norm_(h.parameters(), config.TAPPFL_CLIP)
                opt_h.step()
                tot += float(loss.item())
                n += 1
        logger.info(f"  MI pretrain epoch {ep}/{epochs}: mean -jsd_MI={tot / max(n, 1):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config.TAPPFL_LOCAL_EPOCHS = args.local_epochs   # CLI override read by client_train

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = datetime.now().strftime("%d_%B_%H%M%S")
    run_dir = config.TAPPFL_OUTPUT_DIR / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "train.log")
    logger.info(f"TAPPFL run: {run_name}")
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Device: {device}")

    # Data: gender-aware per-client (subject) datasets + centralized held-out val/test, plus a
    # deterministic pooled view of the training subjects (train_eval_ds) for the frozen probes.
    client_datasets, train_eval_ds, val_ds, test_ds, label2idx = build_tappfl_datasets()
    client_ids = sorted(client_datasets.keys())
    logger.info(f"Clients: {len(client_ids)} | train_eval clips: {len(train_eval_ds)} | "
                f"val clips: {len(val_ds)} | test clips: {len(test_ds)}")

    client_loaders = {
        cid: DataLoader(ds, batch_size=config.FL_CLIENT_BATCH_SIZE, shuffle=True,
                        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=False)
        for cid, ds in client_datasets.items()
    }
    all_records = ([r for ds in client_datasets.values() for r in ds.records]
                   + list(val_ds.records) + list(test_ds.records))
    warm_frame_cache(all_records, workers=config.CACHE_WARM_WORKERS)

    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True,
                            persistent_workers=config.NUM_WORKERS > 0)

    # Model: FE backbone + frozen decoder, initialized from a trained checkpoint.
    model = build_model(config.NUM_CLASSES).to(device)
    init_ckpt = find_init_checkpoint(args)
    if init_ckpt.exists():
        ckpt = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Initialized FE/decoder from {init_ckpt} "
                    f"(epoch {ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', float('nan')):.4f})")
    else:
        logger.warning(f"No init checkpoint at {init_ckpt}; starting from pretrained VideoMAE backbone.")
    freeze_decoder(model)

    # Shared privacy net g_Psi and MI estimator h_Omega (never reset, never communicated).
    d_x = 3 * config.TAPPFL_X_POOL ** 2
    g = PrivacyNet(d_z=768, k=config.TAPPFL_K).to(device)
    h = MIEstimator(d_x=d_x, d_z=768).to(device)
    opt_g = torch.optim.AdamW(g.parameters(), lr=config.TAPPFL_ADV_LR, weight_decay=config.WEIGHT_DECAY)
    opt_h = torch.optim.AdamW(h.parameters(), lr=config.TAPPFL_ADV_LR, weight_decay=config.WEIGHT_DECAY)

    # Adversary CE on the true private label (plain CE, as in the original TAPPFL).
    criterion = nn.CrossEntropyLoss()
    # Cross-client gender-balanced replay buffer so the shared adversary keeps a stable boundary.
    buffer = FeatureBuffer(config.TAPPFL_K, config.TAPPFL_BUFFER_CAP)
    pooled_gender = np.array([r["gender"] for ds in client_datasets.values() for r in ds.records])
    logger.info(f"Pooled gender counts: female={int((pooled_gender == 0).sum())}, "
                f"male={int((pooled_gender == 1).sum())}")

    n_per_round = max(1, round(args.client_fraction * len(client_ids)))
    tappfl_cfg = {
        "algorithm": "TAPPFL (original 1:1:1 local minimax; github.com/TAPPFL/TAPPFL)",
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "lambda": args.lam,
        "lambda_meaning": "weight on privacy term (1=pure privacy, 0=pure utility)",
        "client_fraction": args.client_fraction,
        "clients_per_round": n_per_round,
        "num_clients": len(client_ids),
        "fedavg": "uniform",
        "lr_fe": config.TAPPFL_LR_INIT,
        "lr_adv_mi": config.TAPPFL_ADV_LR,
        "grad_clip": config.TAPPFL_CLIP,
        "mi_conditioning": "u fed into MI critic (scalar output)",
        "x_pool": config.TAPPFL_X_POOL,
        "x_dim": d_x,
        "sensitive_attr": "gender",
        "k_classes": config.TAPPFL_K,
        "adversary_buffer_cap": config.TAPPFL_BUFFER_CAP,
        "adv_reset_every": args.adv_reset_every,
        "adv_warmup_steps": args.adv_warmup_steps,
        "privacy_measured_on": "training (seen) clients, subject-grouped probe",
        "utility_measured_by": f"fresh MLP head on frozen z ({config.TAPPFL_UTIL_HEAD_EPOCHS} ep)",
        "mi_pretrain_epochs": args.mi_pretrain_epochs,
        "eval_every": args.eval_every,
        "init_checkpoint": str(init_ckpt),
        "shared_adversary": "single g_Psi/h_Omega updated sequentially across clients with a "
                            "gender-balanced cross-client replay buffer (subject-per-client => "
                            "single-gender shards; deviation from original per-client adversary)",
        "client_batch_size": config.FL_CLIENT_BATCH_SIZE,
        "model": config.MODEL_NAME,
        "num_classes": config.NUM_CLASSES,
        "seed": config.SEED,
        "val_clips": len(val_ds),
        "test_clips": len(test_ds),
    }
    with open(run_dir / "tappfl_config.json", "w") as f:
        json.dump(tappfl_cfg, f, indent=2)
    logger.info(f"Config: {json.dumps(tappfl_cfg)}")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project="ndv-action-tappfl", name=run_name, config=tappfl_cfg)
        # Use the federated round as the x-axis for every panel (instead of the internal
        # _step counter, which otherwise leaves panels showing "no data for _step").
        wandb.define_metric("round")
        wandb.define_metric("*", step_metric="round")

    if args.mi_pretrain_epochs > 0:
        logger.info(f"Centralized MI pretraining for {args.mi_pretrain_epochs} epochs...")
        mi_pretrain(model, client_loaders, h, opt_h, args.mi_pretrain_epochs, device, logger)

    # Checkpoint selection: maximize (utility - gender-leakage) at eval rounds. val_attr_acc
    # (shared adversary on the val set) is kept only as a cheap per-round monitor.
    best_score = -float("inf")
    best_val_attr_acc = float("inf")
    history = []
    sample_rng = random.Random(config.SEED)

    logger.info(f"Starting TAPPFL: {args.rounds} rounds x {n_per_round}/{len(client_ids)} "
                f"clients/round | lambda={args.lam}")

    for rnd in tqdm(range(1, args.rounds + 1), desc="TAPPFL rounds"):
        selected = (client_ids if n_per_round >= len(client_ids)
                    else sample_rng.sample(client_ids, n_per_round))

        # Relaunch the adversary: a fresh g_Psi, re-trained to competence on the current
        # representation, so the encoder must beat a NEW attacker (not overfit one stale g).
        if args.adv_reset_every > 0 and buffer.ready() and (rnd - 1) % args.adv_reset_every == 0:
            g, opt_g = reinit_adversary(device)
            wf, wl = warmup_adversary(g, opt_g, buffer, criterion, args.adv_warmup_steps)
            logger.info(f"  relaunched adversary g_Psi (warmup CE {wf:.3f} -> {wl:.3f} "
                        f"over {args.adv_warmup_steps} steps)")

        global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        client_stats = fedavg_aggregate(
            model, global_state, client_loaders, selected,
            g, h, opt_g, opt_h, args.lam, device, criterion, buffer, logger,
        )
        mean_total = float(np.mean([s["L_total"] for s in client_stats.values()]))
        mean_jsd = float(np.mean([s["jsd_MI"] for s in client_stats.values()]))
        mean_adv = float(np.mean([s["adv_train_acc"] for s in client_stats.values()]))

        val_metrics = validate_attr_acc(model, g, val_loader, device)
        val_attr_acc = val_metrics["attr_acc"]
        best_val_attr_acc = min(best_val_attr_acc, val_attr_acc)
        logger.info(f"Round {rnd}/{args.rounds} | L_total={mean_total:.4f} | "
                    f"jsd_MI={mean_jsd:.4f} | adv_acc={mean_adv:.3f} | "
                    f"val_attr_acc={val_attr_acc:.4f} (chance={1.0 / config.TAPPFL_K:.3f})")

        round_rec = {"round": rnd, "L_total": mean_total, "jsd_MI": mean_jsd,
                     "adv_train_acc": mean_adv, "val_attr_acc": val_attr_acc,
                     "val_attr_balanced_acc": val_metrics["attr_balanced_acc"]}

        if use_wandb:
            log_round_diagnostics(rnd, client_stats, val_metrics, best_val_attr_acc, args.lam, device)

        # Periodic frozen-encoder eval: gender leakage on SEEN clients + retrained-head utility.
        do_eval = args.eval_every > 0 and (rnd % args.eval_every == 0 or rnd == args.rounds)
        ckpt = {"epoch": rnd, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {}, "val_acc": val_attr_acc,
                "g_state_dict": g.state_dict(), "h_state_dict": h.state_dict()}
        torch.save(ckpt, ckpt_dir / "last_model.pt")

        if do_eval:
            privacy, utility = run_eval(model, train_eval_ds, test_ds, device, config.SEED)
            score = utility["accuracy"] - privacy["probe_balanced_accuracy"]   # high util, low leakage
            logger.info(f"  [eval r{rnd}] seen-client gender leakage: "
                        f"acc={privacy['probe_accuracy']:.4f} bal_acc={privacy['probe_balanced_accuracy']:.4f} "
                        f"(majority={privacy['majority_baseline']:.3f}) | "
                        f"utility(retrained head)={utility['accuracy']:.4f} | score={score:.4f}")
            round_rec.update({"privacy_bal_acc": privacy["probe_balanced_accuracy"],
                              "utility_accuracy": utility["accuracy"], "score": score})
            if use_wandb:
                wandb.log({"round": rnd,
                           "privacy/seen_probe_accuracy": privacy["probe_accuracy"],
                           "privacy/seen_probe_balanced_accuracy": privacy["probe_balanced_accuracy"],
                           "privacy/seen_probe_minus_majority":
                               privacy["probe_accuracy"] - privacy["majority_baseline"],
                           "utility/retrained_head_accuracy": utility["accuracy"],
                           "score/util_minus_leakage": score})
            if score > best_score:
                best_score = score
                ckpt["val_acc"] = privacy["probe_balanced_accuracy"]
                torch.save(ckpt, ckpt_dir / "best_model.pt")
                logger.info(f"  ** New best score={best_score:.4f} "
                            f"(util={utility['accuracy']:.3f}, leakage={privacy['probe_balanced_accuracy']:.3f})"
                            f" — checkpoint saved")

        history.append(round_rec)

    with open(run_dir / "round_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best score (util - leakage)={best_score:.4f}")

    # Load best global FE (best privacy/utility trade-off) for the final report.
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (round {best_ckpt['epoch']}) for evaluation")
    model.eval()

    # ── Final eval: seen-client gender leakage + retrained-head utility ──
    privacy, utility = run_eval(model, train_eval_ds, test_ds, device, config.SEED)
    logger.info(f"Privacy (seen-client gender probe, subject-grouped): "
                f"acc={privacy['probe_accuracy']:.4f} bal_acc={privacy['probe_balanced_accuracy']:.4f} "
                f"(majority={privacy['majority_baseline']:.3f}, chance={privacy['chance']:.3f})")
    if privacy["note"]:
        logger.warning(f"  Privacy probe caveat: {privacy['note']}")
    logger.info(f"Utility (retrained head) test accuracy: {utility['accuracy']:.4f}")
    y_true, y_pred = np.array(utility["test_true"]), np.array(utility["test_pred"])
    logger.info("\n" + compute_metrics(y_true, y_pred, list(config.LABEL_NAMES))["report"])

    metrics = {
        "privacy": privacy,
        "utility": {k: v for k, v in utility.items() if k not in ("test_pred", "test_true")},
        "best_score_util_minus_leakage": best_score,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(y_true, y_pred, list(config.LABEL_NAMES), run_dir / "confusion_matrix.png")

    if use_wandb:
        wandb.summary["final/seen_probe_accuracy"] = privacy["probe_accuracy"]
        wandb.summary["final/seen_probe_balanced_accuracy"] = privacy["probe_balanced_accuracy"]
        wandb.summary["final/utility_retrained_head_accuracy"] = utility["accuracy"]
        wandb.summary["final/best_score"] = best_score
        wandb.log({"final/confusion_matrix": wandb.plot.confusion_matrix(
            preds=y_pred.tolist(), y_true=y_true.tolist(), class_names=list(config.LABEL_NAMES))})
        wandb.finish()

    logger.info(f"For the full attribute-inference analysis, run:\n"
                f"  python -m src.evaluation.attack_inference --checkpoint {best_path}")
    logger.info(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
