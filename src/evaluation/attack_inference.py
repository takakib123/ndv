"""
Attribute Inference Attack: predict participant gender from VideoMAE representations.

Design (PARTICIPANT-BASED probe):
  1. Load trained VideoMAE checkpoint; hook the backbone to capture features.
  2. Run EVERY action clip through the model (no_grad) -> (N_clips, 768) features.
     Each participant contributes ~8 action clips; we keep them as individual
     probe samples (we do NOT mean-pool to one vector per participant).
  3. Tag each clip with its gender label and its model split (train / val / test).
  4. Two independent probe experiments:
        - SEEN     : clips from the model's training-split participants.
        - HELD-OUT : clips from the model's val + test participants.
     Comparing the two tells us whether gender is decodable for identities the
     model saw during training versus identities it never saw.
  5. Each experiment runs a logistic-regression probe (the same attacker used in-run by
     tappfl_train.probe_gender_seen) under LEAVE-ONE-PARTICIPANT-OUT: for each person,
     train on all the others and predict theirs. Strongest attacker, deterministic, and
     every participant is tested exactly once. Uncertainty = bootstrap CI over
     PARTICIPANTS; a weak-attacker split is reported alongside for context.
  6. Save plots (ROC, confusion matrices, t-SNE) + JSON.

Why participant-grouped: gender is constant per person, so a clip-level split lets the
probe memorize IDENTITY and read gender off it -- inflating leakage (0.885 clip-level vs
0.616 participant-level on the same features). Only the grouped number answers "does the
representation leak gender for people the attacker hasn't profiled?". It also means the
effective sample size is the participant count (tens), not the clip count -- hence the
repeats and the per-participant vote.
"""

import argparse
import hashlib
import json
import logging
import random
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.data.dataset import ActionVideoDataset, load_filtered_df, _split_subjects
from src.data.redaction import parse_condition, cache_tag as redaction_cache_tag
from src.model import build_model

ATTACK_DIR = config.OUTPUT_DIR / "attack"
FEATURE_CACHE_DIR = ATTACK_DIR / "feature_cache"   # stable, reused across runs (keyed by checkpoint)
GENDER_CSV = config.GENDER_CSV
EXTRACT_BATCH_SIZE = 8

logger = logging.getLogger("attack")


def parse_args():
    p = argparse.ArgumentParser(description="Attribute inference (gender) attack on a VideoMAE checkpoint")
    p.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR / "best_model.pt",
                   help="Path to the trained model checkpoint (.pt) to attack, or the literal "
                        "'pretrained' to attack the un-fine-tuned encoder (leakage floor)")
    p.add_argument("--redaction", default="none",
                   help="Redaction applied to every clip before the encoder sees it, e.g. "
                        "face_blur_s0.5 to attack a blur-trained model on the inputs it was "
                        "trained on. 'none' attacks clean video.")
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute clip features even if a cached copy exists")
    return p.parse_args()


def setup_logger(log_path):
    """Logger that writes timestamped, leveled messages to both a file and the console."""
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SubjectAwareVideoDataset(ActionVideoDataset):
    """ActionVideoDataset that also returns the participant subject ID per clip."""

    def __init__(self, df: pd.DataFrame, label2idx: dict, augment: bool = False, redactor=None):
        super().__init__(df, label2idx, augment, redactor=redactor)

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        item["subject"] = self.records[idx]["subject"]
        return item


def build_subject_datasets(redactor=None):
    """Replicate build_datasets() split; return all clips as one dataset plus split sets.

    `redactor` optionally occludes a region of every clip before it reaches the encoder --
    that is what src.evaluation.redaction_leakage uses to measure how much leakage survives
    removing the visible evidence.
    """
    df = load_filtered_df()
    train_subjects, val_subjects, test_subjects = _split_subjects(df, config.SEED)

    label2idx = {label: i for i, label in enumerate(config.LABEL_NAMES)}
    all_ds = SubjectAwareVideoDataset(df, label2idx, augment=False, redactor=redactor)

    return all_ds, train_subjects, val_subjects, test_subjects


def feature_cache_key(ckpt_path, redaction_key: str = "none") -> str:
    """Cache id for extracted clip features.

    Keyed on the checkpoint's path AND mtime so a retrained checkpoint never reuses stale
    features, and on the redaction condition so redacted and clean features never collide.
    """
    # "-qvbias": pretrained features from before the q_bias/v_bias load fix are stale.
    stamp = "pretrained-qvbias" if ckpt_path is None else f"{ckpt_path}|{ckpt_path.stat().st_mtime_ns}"
    return hashlib.md5(f"{stamp}|{redaction_key}".encode()).hexdigest()[:12]


def load_encoder(ckpt_path, device):
    """Load a VideoMAE checkpoint, or the un-fine-tuned pretrained encoder.

    `ckpt_path=None` gives the pretrained model untouched, which is the data-intrinsic
    leakage floor: whatever a probe recovers from it was never learned from this dataset.
    """
    model = build_model(config.NUM_CLASSES)
    if ckpt_path is None:
        logger.info("Using PRETRAINED encoder (no fine-tuning) — data-intrinsic leakage floor")
        return model.to(device).eval(), {"epoch": -1, "val_acc": float("nan")}

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    return model.to(device).eval(), ckpt


# ---------------------------------------------------------------------------
# Per-clip feature extraction
# ---------------------------------------------------------------------------

def extract_clip_features(model, all_ds, device):
    """
    Run every clip through the VideoMAE backbone (forward hook on model.videomae),
    mean-pool over patch tokens -> one 768-d vector PER CLIP.

    Returns: X (N_clips, 768), subjects (N_clips,), label_idx (N_clips,).
    """
    captured = {}

    def hook(module, inp, out):
        captured["feat"] = out.last_hidden_state.mean(dim=1).cpu().float()

    handle = model.videomae.register_forward_hook(hook)

    loader = DataLoader(
        all_ds,
        batch_size=EXTRACT_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=config.NUM_WORKERS > 0,
    )

    clip_feats, clip_subjects, clip_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting per-clip backbone features"):
            pixel_values = batch["pixel_values"].to(device)
            with torch.amp.autocast("cuda", enabled=config.AMP and device.type == "cuda"):
                model(pixel_values=pixel_values)
            clip_feats.append(captured["feat"])
            clip_subjects.extend(batch["subject"])
            clip_labels.extend(batch["labels"].tolist())

    handle.remove()

    X = torch.cat(clip_feats, dim=0).numpy()
    subjects = np.array(clip_subjects)
    labels = np.array(clip_labels)
    logger.info(f"Extracted {X.shape[0]} clip features (dim={X.shape[1]}) "
          f"from {len(set(clip_subjects))} participants")
    return X, subjects, labels


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

# Logistic-regression probe -- the SAME attacker the in-run probe in tappfl_train.py uses,
# so training-time and offline leakage numbers are directly comparable. Name -> C (inverse
# L2 strength). An MLP probe was tried and, once identity leakage is removed by the
# participant-grouped split, it recovers no more gender than LR (0.616 vs 0.638 on the same
# features) -- so the simpler, faster, deterministic probe is used.
PROBES = {"LogReg": 1.0}

# HEADLINE metric = leave-one-participant-out (LOPO): train on all other participants, test on
# the held-out one. Privacy is a WORST-CASE property, so the attacker gets the largest possible
# training set. LOPO is also deterministic (no split randomness) and tests every participant
# exactly once. Attacker budget dominates the measurement -- on the same undefended encoder:
#   11 train participants -> 0.574 | 26 -> 0.647 | 52 (LOPO) -> 0.741 balanced accuracy.
# So starving the attacker understates leakage; LOPO is the number to quote.
#
# SECONDARY (weak-attacker sanity check): train on PROBE_TRAIN_FRAC of participants, test on
# the rest, repeated. Reported for context only -- it systematically understates leakage.
PROBE_TRAIN_FRAC = 0.20
N_SPLITS = int(round(1 / PROBE_TRAIN_FRAC))
N_REPEATS = 10        # repeats with different participant partitions (weak-attacker check)
N_BOOTSTRAP = 2000    # bootstrap resamples for the participant-level CI on the LOPO estimate


def _make_probe(C: float) -> LogisticRegression:
    """Same probe as tappfl_train.probe_gender_seen, so the two agree."""
    return LogisticRegression(C=C, max_iter=1000, class_weight="balanced")


def _bootstrap_ci_participants(y, pred, subjects, n_boot=N_BOOTSTRAP, seed=config.SEED):
    """95% CI on balanced accuracy, resampling PARTICIPANTS with replacement. Participants are
    the effective sample unit (gender is constant per person), so a clip-level CI would be far
    too optimistic. If the CI excludes 0.50, the representation demonstrably leaks."""
    rng = np.random.default_rng(seed)
    subs = np.unique(subjects)
    idx_by_sub = {s: np.where(subjects == s)[0] for s in subs}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([idx_by_sub[s] for s in pick])
        if len(np.unique(y[idx])) > 1:
            boots.append(balanced_accuracy_score(y[idx], pred[idx]))
    if not boots:
        return None, None
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def _lopo_predictions(X, y, subjects, C):
    """Leave-one-participant-out: for each participant, train on ALL others and predict theirs.
    Every clip gets exactly one out-of-fold prediction from the strongest possible attacker."""
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y), dtype=float)
    tr_accs = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=subjects):
        sc = StandardScaler().fit(X[tr])
        clf = _make_probe(C).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
        prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        tr_accs.append(clf.score(sc.transform(X[tr]), y[tr]))
    return pred, prob, float(np.mean(tr_accs))


def matched_budget_lopo(X, y, subjects, n_participants, C=1.0, n_draws=None, seed=config.SEED):
    """LOPO balanced accuracy with the attacker restricted to `n_participants` people.

    SEEN has 53 participants and HELD-OUT 24, so a plain LOPO comparison hands the SEEN
    attacker 52 training participants against the HELD-OUT attacker's 23. Attacker budget
    dominates this measurement (11 -> .574, 26 -> .647, 52 -> .741 on the same undefended
    encoder), so the raw SEEN/HELD-OUT pair is not comparable and the memorization gap read
    off it is meaningless -- indeed HELD-OUT currently scores ABOVE SEEN.

    Here each draw subsamples `n_participants` people, runs LOPO within that subset, and the
    draws are averaged. Both groups then face an attacker of identical strength, so any
    remaining difference is attributable to training-set membership.

    Returns (mean_bal_acc, std_over_draws, pooled_pred, pooled_y, pooled_subjects) where the
    pooled arrays come from the first draw and back the bootstrap CI.
    """
    n_draws = n_draws or config.MATCHED_BUDGET_DRAWS
    subs = np.unique(subjects)
    if len(subs) <= n_participants:
        pred, prob, _ = _lopo_predictions(X, y, subjects, C)
        return (float(balanced_accuracy_score(y, pred)), 0.0, pred, y, subjects)

    rng = np.random.default_rng(seed)
    sub_gender = {s: y[subjects == s][0] for s in subs}
    accs, first = [], None
    for d in range(n_draws):
        # Stratify by gender so a draw cannot land single-class and waste itself.
        males = [s for s in subs if sub_gender[s] == 1]
        females = [s for s in subs if sub_gender[s] == 0]
        n_m = max(1, round(n_participants * len(males) / len(subs)))
        n_f = max(1, n_participants - n_m)
        pick = np.concatenate([
            rng.choice(males, min(n_m, len(males)), replace=False),
            rng.choice(females, min(n_f, len(females)), replace=False),
        ])
        mask = np.isin(subjects, pick)
        if len(np.unique(y[mask])) < 2:
            continue
        pred, _, _ = _lopo_predictions(X[mask], y[mask], subjects[mask], C)
        accs.append(balanced_accuracy_score(y[mask], pred))
        if first is None:
            first = (pred, y[mask], subjects[mask])

    if not accs:
        return (float("nan"), float("nan"), np.array([]), np.array([]), np.array([]))
    return (float(np.mean(accs)), float(np.std(accs)), *first)


def _weak_attacker_bal_acc(X, y, subjects, C):
    """Secondary check: train on PROBE_TRAIN_FRAC of participants, score on the rest, repeated.
    Returns (mean, std) balanced accuracy over repeats, or (None, None)."""
    per_repeat = []
    for r in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.SEED + r)
        fold = []
        for big, small in sgkf.split(X, y, groups=subjects):
            tr, te = small, big          # train on the SMALL fold -> weak attacker
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = _make_probe(C).fit(sc.transform(X[tr]), y[tr])
            fold.append(balanced_accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
        if fold:
            per_repeat.append(float(np.mean(fold)))
    if not per_repeat:
        return None, None
    return float(np.mean(per_repeat)), float(np.std(per_repeat))


def _eval_metrics(y_te, y_pred, y_prob):
    has_both = len(np.unique(y_te)) > 1
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "auc_roc": float(roc_auc_score(y_te, y_prob)) if has_both else None,
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist(),
        "y_true": y_te.tolist(),
        "y_prob": y_prob.tolist(),
    }


def _subject_vote_accuracy(y, pred, subjects):
    """Per-PARTICIPANT accuracy: majority-vote each person's clip predictions against their
    true gender. Participants are the effective sample unit (gender is constant per person)."""
    subs = np.unique(subjects)
    correct = [int((pred[subjects == s].mean() > 0.5) == bool(y[subjects == s][0])) for s in subs]
    return float(np.mean(correct)), int(np.sum(correct)), len(subs)


def run_probe_group(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, group_name: str,
                    matched_n: int = None):
    """
    PARTICIPANT-BASED attribute-inference probe.

    Every clip of a participant stays in the same fold, so the probe is always tested on
    people it has never seen. This measures whether GENDER generalizes across identities --
    the actual attribute-leakage question. A clip-level split instead lets the probe memorize
    identity and badly inflates leakage (measured on the same features: 0.885 clip-level vs
    0.616 participant-level).

    HEADLINE = leave-one-participant-out: for each participant, train on ALL others and predict
    theirs. Privacy is worst-case, so the attacker gets the biggest training set it can; LOPO is
    also deterministic and tests every participant exactly once. Uncertainty is a bootstrap CI
    over PARTICIPANTS (the effective sample unit) -- if the CI excludes 0.50, it demonstrably
    leaks. A weak-attacker check (train on PROBE_TRAIN_FRAC of participants) is also reported,
    since attacker budget dominates: 11 train -> 0.574, 26 -> 0.647, 52 (LOPO) -> 0.741.
    """
    n = len(y)
    n_participants = len(set(subjects))

    logger.info(f"\n{'='*64}")
    logger.info(f"  GROUP: {group_name}")
    logger.info(f"  {n_participants} participants  |  {n} clips "
                f"(male={int(y.sum())}, female={int((y == 0).sum())})")
    logger.info(f"  headline: leave-one-participant-out ({n_participants - 1} train / 1 test, "
                f"x{n_participants} folds, deterministic)")
    logger.info(f"{'='*64}")

    results = {}
    for clf_name, C in PROBES.items():
        logger.info(f"\n  [{clf_name}]  C={C}")

        # ── HEADLINE: leave-one-participant-out ──
        pred, prob, tr_acc = _lopo_predictions(X, y, subjects, C)
        m = _eval_metrics(y, pred, prob)
        lo, hi = _bootstrap_ci_participants(y, pred, subjects)
        vote_acc, vote_n, vote_tot = _subject_vote_accuracy(y, pred, subjects)
        majority = float(max(y.mean(), 1 - y.mean()))
        significant = bool(lo is not None and lo > 0.5)

        # ── SECONDARY: weak attacker (starved training set) ──
        weak_ba, weak_sd = _weak_attacker_bal_acc(X, y, subjects, C)

        # ── MATCHED BUDGET: the only SEEN-vs-HELD-OUT comparison that means anything ──
        matched_ba = matched_sd = None
        if matched_n is not None:
            matched_ba, matched_sd, *_ = matched_budget_lopo(X, y, subjects, matched_n, C)

        auc_s = f"{m['auc_roc']:.3f}" if m["auc_roc"] is not None else "nan"
        ci_s = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "[n/a]"
        logger.info(f"    train acc            : {tr_acc:.3f}")
        logger.info(f"    LOPO acc             : {m['accuracy']:.3f}")
        logger.info(f"    LOPO bal-acc         : {m['balanced_accuracy']:.3f}  95%CI {ci_s}   "
                    f"(majority={majority:.3f}, chance=0.500)")
        logger.info(f"    LOPO AUC             : {auc_s}   F1={m['f1_macro']:.3f}")
        logger.info(f"    per-participant vote : {vote_n}/{vote_tot} correct ({vote_acc:.3f})")
        logger.info(f"    -> {'LEAKS (CI excludes chance)' if significant else 'n.s. (CI covers chance)'}")
        if weak_ba is not None:
            logger.info(f"    [weak attacker, {int(PROBE_TRAIN_FRAC*100)}% of participants: "
                        f"bal-acc={weak_ba:.3f} +/- {weak_sd:.3f} -- understates leakage]")
        if matched_ba is not None:
            logger.info(f"    matched budget ({matched_n} participants): "
                        f"bal-acc={matched_ba:.3f} +/- {matched_sd:.3f}  "
                        f"<- use THIS to compare groups")

        results[clf_name] = {
            **m,
            "ci95_low": lo,
            "ci95_high": hi,
            "significant_vs_chance": significant,
            "best_params": {"C": C},
            "train_accuracy": tr_acc,
            "loss_curve": [],            # LogisticRegression has no per-epoch curve
            "validation_scores": [],
            "n_participants": n_participants,
            "n_clips": n,
            "per_participant_vote_accuracy": vote_acc,
            "weak_attacker_bal_acc": weak_ba,
            "weak_attacker_bal_acc_std": weak_sd,
            "weak_attacker_train_frac": PROBE_TRAIN_FRAC,
            "matched_budget_bal_acc": matched_ba,
            "matched_budget_bal_acc_std": matched_sd,
            "matched_budget_n_participants": matched_n,
            "split": "leave-one-participant-out (headline); fractional participant split "
                     "reported as a weak-attacker check",
            "majority_baseline": majority,
        }

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curves(group_results: dict, save_path: Path):
    fig, axes = plt.subplots(1, len(group_results), figsize=(7 * len(group_results), 6))
    if len(group_results) == 1:
        axes = [axes]
    for ax, (group, results) in zip(axes, group_results.items()):
        for clf_name, res in results.items():
            if res["y_true"] is None or res["y_prob"] is None:
                continue
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
            ax.plot(fpr, tpr, lw=2, label=f"{clf_name} (AUC={res['auc_roc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (0.500)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Gender Inference — {group}\n(participant-grouped CV, pooled out-of-fold)")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC curves -> {save_path}")


def plot_confusion_matrices(group_results: dict, save_path: Path):
    rows = list(group_results.items())
    n_clf = len(PROBES)
    fig, axes = plt.subplots(len(rows), n_clf, figsize=(5 * n_clf, 4 * len(rows)),
                             squeeze=False)
    labels = ["Female", "Male"]
    for r, (group, results) in enumerate(rows):
        for c, (clf_name, res) in enumerate(results.items()):
            ax = axes[r, c]
            cm = np.array(res["confusion_matrix"])
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            annot = np.array([[f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
                               for j in range(cm.shape[1])]
                              for i in range(cm.shape[0])])
            sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                        xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
            auc_str = f"{res['auc_roc']:.3f}" if res["auc_roc"] else "N/A"
            ax.set_title(f"{group} — {clf_name}\nacc={res['accuracy']:.3f}  AUC={auc_str}")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.suptitle("Gender Inference — Confusion Matrices (probe-test clips)", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrices -> {save_path}")


def plot_mlp_training_curves(group_results: dict, save_path: Path):
    """Per-epoch MLP training loss (and held-out validation score) per group."""
    groups = list(group_results.items())
    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5),
                             squeeze=False)
    for ax, (group, results) in zip(axes[0], groups):
        res = results.get("MLP-3layer", {})
        loss = res.get("loss_curve", [])
        val = res.get("validation_scores", [])
        if not loss:
            ax.set_visible(False)
            continue
        epochs = np.arange(1, len(loss) + 1)
        l1 = ax.plot(epochs, loss, color="#e74c3c", lw=2, label="Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss", color="#e74c3c")
        ax.tick_params(axis="y", labelcolor="#e74c3c")
        lines = l1
        if val:
            ax2 = ax.twinx()
            l2 = ax2.plot(np.arange(1, len(val) + 1), val, color="#3498db",
                          lw=2, label="Validation score")
            ax2.set_ylabel("Validation score", color="#3498db")
            ax2.tick_params(axis="y", labelcolor="#3498db")
            lines = l1 + l2
        ax.set_title(f"MLP probe training curve — {group}\n"
                     f"(best alpha={res.get('best_params', {}).get('alpha', 'N/A')})")
        ax.legend(lines, [ln.get_label() for ln in lines], loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"MLP training curves -> {save_path}")


def plot_tsne(X: np.ndarray, y: np.ndarray, group_flag: np.ndarray, save_path: Path):
    """t-SNE of all clips, coloured by gender, marker by seen/held-out."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=config.SEED, max_iter=1000)
    X_2d = tsne.fit_transform(X)
    palette = {0: "#e74c3c", 1: "#3498db"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for flag, marker, lbl in [(1, "o", "Seen (train split)"), (0, "^", "Held-out (val+test)")]:
        mask = group_flag == flag
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[palette[yi] for yi in y[mask]], marker=marker, s=45,
                   alpha=0.7, edgecolors="white", linewidths=0.4, label=lbl)
    handles = [
        mpatches.Patch(facecolor="#e74c3c", label="Female"),
        mpatches.Patch(facecolor="#3498db", label="Male"),
        plt.Line2D([], [], marker="o", color="gray", linestyle="", label="Seen"),
        plt.Line2D([], [], marker="^", color="gray", linestyle="", label="Held-out"),
    ]
    ax.legend(handles=handles)
    ax.set_title("t-SNE of per-clip VideoMAE features (768-d -> 2-d)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"t-SNE plot -> {save_path}")


def print_summary(group_results: dict):
    width = 78
    logger.info("\n" + "=" * width)
    logger.info("ATTRIBUTE INFERENCE ATTACK — SUMMARY (participant-grouped logistic probe)")
    logger.info("=" * width)
    logger.info("NOTE: participant-grouped CV — a person's clips never span train/test, so")
    logger.info("      these measure gender generalization, not identity memorization.")
    logger.info("      Effective n = participants, not clips; see per-participant vote.")
    hdr = (f"  {'Group':<20}{'Probe':<10}{'BalAcc':>8}{'95% CI':>17}"
           f"{'AUC':>7}{'Base':>7}{'Vote':>7}  Leaks?")
    logger.info("\n" + hdr)
    logger.info("  " + "-" * (len(hdr) - 2))
    for group, results in group_results.items():
        for clf_name, res in results.items():
            auc = res["auc_roc"] if res["auc_roc"] is not None else float("nan")
            ci = (f"[{res['ci95_low']:.3f},{res['ci95_high']:.3f}]"
                  if res.get("ci95_low") is not None else "[n/a]")
            leaks = "YES" if res.get("significant_vs_chance") else "n.s."
            logger.info(f"  {group:<20}{clf_name:<10}{res['balanced_accuracy']:8.3f}{ci:>17}"
                  f"{auc:7.3f}{res['majority_baseline']:7.3f}"
                  f"{res['per_participant_vote_accuracy']:7.3f}  {leaks}")
    logger.info("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    pretrained = str(args.checkpoint) == "pretrained"
    ckpt_path = None if pretrained else args.checkpoint.resolve()
    if ckpt_path is not None and not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")

    redactor = parse_condition(args.redaction, seed=config.SEED)

    # Timestamped run folder (one self-contained folder per attack run)
    run_name = datetime.now().strftime("%d_%B_%H%M%S")
    if not redactor.is_identity:
        run_name += f"_{redactor.key()}"
    run_dir = ATTACK_DIR / "results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    setup_logger(run_dir / "run.log")
    logger.info(f"Run: {run_name}")
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Attack-input redaction: {redactor.key()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, ckpt = load_encoder(ckpt_path, device)

    # Run config (saved locally so the attack run is reproducible)
    run_cfg = {
        "checkpoint": "pretrained" if pretrained else str(ckpt_path),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_val_acc": float(ckpt["val_acc"]),
        "model": config.MODEL_NAME,
        "redaction": redactor.key(),
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "probe_cv": f"leave-one-participant-out (headline) + {int(PROBE_TRAIN_FRAC*100)}% weak-attacker check",
        "seed": config.SEED,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)
    logger.info(f"Config: {json.dumps(run_cfg)}")

    # Dataset + split membership
    all_ds, train_subjects, val_subjects, test_subjects = build_subject_datasets(redactor=redactor)
    logger.info(f"Total clips: {len(all_ds)}  |  "
          f"train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)} subjects")

    # Extract per-clip backbone features. Cache is keyed by checkpoint (path + mtime)
    # so a different/retrained checkpoint never reuses stale features.
    # Same key scheme as redaction_leakage, so the two scripts share cached features.
    rkey = redactor.key() if redactor.is_identity else f"{redactor.key()}|{redaction_cache_tag()}"
    ckpt_key = feature_cache_key(ckpt_path, rkey)
    cache = FEATURE_CACHE_DIR / f"clip_features_{ckpt_key}.npz"
    if cache.exists() and not args.no_cache:
        data = np.load(cache, allow_pickle=True)
        X_all, subj_all, lbl_all = data["X"], data["subjects"], data["labels"]
        logger.info(f"Loaded {X_all.shape[0]} cached clip features from {cache}")
    else:
        X_all, subj_all, lbl_all = extract_clip_features(model, all_ds, device)
        np.savez(cache, X=X_all, subjects=subj_all, labels=lbl_all)
        logger.info(f"Clip features cached -> {cache}")

    # Gender labels
    gender_df = pd.read_csv(GENDER_CSV)
    gender_df.columns = gender_df.columns.str.strip()
    gender_df = gender_df.dropna(subset=["Subject", "Gender"])
    gender_df["Subject"] = gender_df["Subject"].str.strip()
    gender_df["Gender"] = gender_df["Gender"].str.strip()
    gender_map = dict(zip(gender_df["Subject"], gender_df["Gender"]))

    # Keep only clips whose participant has a known gender
    has_gender = np.array([s in gender_map for s in subj_all])
    X_all, subj_all, lbl_all = X_all[has_gender], subj_all[has_gender], lbl_all[has_gender]
    y_all = np.array([1 if gender_map[s] == "male" else 0 for s in subj_all])

    seen_mask = np.array([s in train_subjects for s in subj_all])
    heldout_mask = np.array([s in (val_subjects | test_subjects) for s in subj_all])

    logger.info(f"\nClips with known gender: {len(y_all)}  "
          f"(seen={int(seen_mask.sum())}, held-out={int(heldout_mask.sum())})")

    # Two independent probe experiments. The matched budget is the smaller group's participant
    # count, so both attackers train on the same number of people and the SEEN-vs-HELD-OUT
    # difference reflects training-set membership rather than attacker strength.
    matched_n = min(len(set(subj_all[seen_mask])), len(set(subj_all[heldout_mask])))
    group_results = {}
    group_results["SEEN (train split)"] = run_probe_group(
        X_all[seen_mask], y_all[seen_mask], subj_all[seen_mask], "SEEN (train split)", matched_n)
    group_results["HELD-OUT (val+test)"] = run_probe_group(
        X_all[heldout_mask], y_all[heldout_mask], subj_all[heldout_mask], "HELD-OUT (val+test)",
        matched_n)

    # Save JSON (omit raw per-sample arrays)
    json_out = {
        "split_note": "Leave-one-participant-out (headline): train on all other participants, "
                      "predict the held-out one -- strongest attacker, deterministic, no "
                      "identity leakage. 95% CI bootstrapped over participants; if it "
                      "excludes 0.50 the representation demonstrably leaks. A weak-attacker "
                      "fractional split is reported alongside and understates leakage.",
        "groups": {
            group: {
                clf: {k: v for k, v in res.items()
                      if k not in ("y_true", "y_prob", "confusion_matrix")}
                for clf, res in results.items()
            }
            for group, results in group_results.items()
        },
    }
    out_json = run_dir / "attack_results.json"
    with open(out_json, "w") as f:
        json.dump(json_out, f, indent=2)
    logger.info(f"Results JSON -> {out_json}")

    # Plots
    plot_roc_curves(group_results, run_dir / "roc_curves.png")
    plot_confusion_matrices(group_results, run_dir / "confusion_matrices.png")
    if any(res.get("loss_curve") for r in group_results.values() for res in r.values()):
        plot_mlp_training_curves(group_results, run_dir / "mlp_training_curves.png")
    group_flag = seen_mask.astype(int)  # 1 = seen, 0 = held-out
    plot_tsne(X_all, y_all, group_flag, run_dir / "feature_tsne.png")

    print_summary(group_results)
    logger.info(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
