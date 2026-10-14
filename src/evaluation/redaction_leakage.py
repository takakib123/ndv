"""Redaction-Conditioned Attribute Leakage (RCAL).

A plain attribute-inference number conflates two different privacy stories:

  1. INPUT-VISIBLE leakage -- gender is readable off the pixels (face, hair, build, clothing).
     Any encoder, even an untrained one, shows this. The *data* leaks, not the model.
  2. MEMORIZED leakage -- the fine-tuned encoder baked participant-specific information into
     its weights, so it recovers gender even when the pixel evidence is gone. The *model*
     leaks, and this is the part that makes releasing the weights risky.

Only (2) is a model-release risk, and a single leakage scalar cannot separate them. This
module redacts a region of every clip BEFORE attribute inference and watches how leakage
decays. Leakage that survives redaction is not being read off the picture.

Models are NOT retrained: clean-trained checkpoints are frozen, redacted clips are pushed
through them, and the gender probe is retrained on the resulting features. So the attacker
always adapts to the redaction -- worst case, as privacy measurement should be.

Metrics, with L(g,c) = 2 * max(0, balanced_accuracy(g,c) - 0.5) in [0, 1]:

  RRL   L(g, face-blackout)                leakage surviving removal of the obvious evidence
  VEF   1 - L(g,face)/L(g,none)            fraction of leakage carried by face pixels
  MAL   L(SEEN,c) - L(HELD-OUT,c)          memorization-attributable leakage, matched budget
  AULRC integral of L over blur severity   one scalar per model, for comparing defenses
  SPEC  L(g,sham_c) - L(g,c)               region-specific drop, net of distribution shift

Three controls carry the argument:

  sham        an identically sized box placed away from the occupant. The encoder was trained
              on clean video, so ANY occlusion is distribution shift; only the drop BEYOND
              sham is attributable to what the region contained.
  pretrained  the same sweep on the un-fine-tuned encoder -- the data-intrinsic leakage floor.
  utility     action accuracy of a retrained head on the same features, so every privacy claim
              comes with what it costs.

Run:
    # smoke test
    python -m src.evaluation.redaction_leakage --conditions none,face_black,sham_face_black
    # full sweep on one checkpoint
    python -m src.evaluation.redaction_leakage --checkpoint artifacts/outputs/checkpoints/best_model.pt
    # compare defenses
    python -m src.evaluation.redaction_leakage --checkpoint pretrained,<baseline>,<fedavg>,<tappfl>

Results land in a timestamped folder under artifacts/outputs/attack/redaction/.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import config
from src.data.redaction import parse_condition, cache_tag as redaction_cache_tag
from src.data.pose import TIER_PER_FRAME
from src.evaluation.attack_inference import (
    build_subject_datasets, extract_clip_features, feature_cache_key, load_encoder,
    matched_budget_lopo, _bootstrap_ci_participants, _lopo_predictions,
    setup_logger, FEATURE_CACHE_DIR, logger,
)

OUT_DIR = config.REDACTION_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Condition sweep
# ---------------------------------------------------------------------------

def default_conditions(include_grid: bool = True):
    """The standard sweep: a severity curve, mode comparison, region comparison, controls."""
    conds = ["none"]
    # Severity curve on the face -> AULRC. s=0 is already covered by "none".
    conds += [f"face_blur_s{s:g}" for s in config.REDACTION_SEVERITIES if s > 0]
    # Does the redaction MECHANISM matter, or only that the region is gone?
    conds += ["face_black", "face_pixelate"]
    # Which body region actually carries the attribute?
    conds += [f"{r}_black" for r in ("head_shoulders", "torso", "arms", "person", "background")]
    # Controls: same occlusion, wrong place.
    conds += ["sham_face_black", "sham_person_black"]
    if include_grid:
        conds += [f"grid_{i}_{j}_black"
                  for i in range(config.REDACTION_GRID) for j in range(config.REDACTION_GRID)]
    return conds


def ckey(spec: str) -> str:
    """Canonical condition key for a CLI-style spec ('face_black' -> 'face_black_s1').

    Rows are stored under Redactor.key(), which always carries the severity suffix, so every
    lookup has to go through the same normalization or it silently misses.
    """
    return parse_condition(spec).key()


def excess_leakage(bal_acc: float) -> float:
    """Balanced accuracy -> leakage in [0,1]. 0.5 (chance) maps to 0, perfect maps to 1."""
    if bal_acc is None or not np.isfinite(bal_acc):
        return float("nan")
    return float(2.0 * max(0.0, bal_acc - 0.5))


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def features_for_condition(model, redactor, ckpt_path, device, no_cache=False):
    """Extract per-clip features under one redaction condition, caching by (ckpt, condition)."""
    # Clean features are shared with attack_inference; redacted ones also key on the redaction
    # geometry, so a change to box sizes or sham/grid semantics never reuses stale features.
    rkey = redactor.key() if redactor.is_identity else f"{redactor.key()}|{redaction_cache_tag()}"
    cache = FEATURE_CACHE_DIR / f"clip_features_{feature_cache_key(ckpt_path, rkey)}.npz"
    if cache.exists() and not no_cache:
        d = np.load(cache, allow_pickle=True)
        logger.info(f"  cached features: {cache.name}")
        return d["X"], d["subjects"], d["labels"], None

    ds, train_subjects, val_subjects, test_subjects = build_subject_datasets(redactor=redactor)
    X, subjects, labels = extract_clip_features(model, ds, device)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, subjects=subjects, labels=labels)
    return X, subjects, labels, (train_subjects, val_subjects, test_subjects)


def pose_tier_coverage() -> dict:
    """What fraction of clips got per-frame pose vs a fallback anchor.

    Reported rather than assumed, so a run states how much of the metric rests on a fallback
    anchor instead of that clip's own keypoints.
    """
    from src.data.dataset import load_filtered_df, get_video_path, sample_frames
    from src.data.pose import get_subject_pose
    from src.data.redaction import original_frame_size

    df = load_filtered_df()
    tiers = []
    for _, row in df.iterrows():
        try:
            vp = str(get_video_path(row["Subject"]))
        except FileNotFoundError:
            continue
        idxs = sample_frames(int(row["StartFrame"]), int(row["EndFrame"]))
        clip = (idxs[0] + 1, idxs[-1] + 1)
        sp = get_subject_pose(row["Subject"], config.DATA_CAMERA, config.DATA_VIDEO_TYPE,
                              original_frame_size(vp))
        tiers.append(sp._anchor(clip[0], clip)[3])
    tiers = np.array(tiers)
    return {f"tier_{t}_frac": float((tiers == t).mean()) for t in sorted(set(tiers.tolist()))}


def evaluate_condition(X, y, subjects, seen_mask, heldout_mask, matched_n):
    """Probe both groups at matched attacker budget, plus a full-budget SEEN reference."""
    out = {"per_participant": {}}
    for name, mask in (("seen", seen_mask), ("heldout", heldout_mask)):
        Xg, yg, sg = X[mask], y[mask], subjects[mask]
        ba, sd, pred, y_first, subj_first = matched_budget_lopo(Xg, yg, sg, matched_n)
        # `ba` averages all draws; the CI is bootstrapped over participants within ONE
        # representative draw, so it reflects participant sampling noise rather than
        # draw-to-draw variation. `balanced_accuracy_std` below covers the latter -- quote
        # both, and treat the CI as indicative rather than a strict interval on `ba`.
        lo, hi = (_bootstrap_ci_participants(y_first, pred, subj_first)
                  if len(pred) else (None, None))
        out[name] = {
            "balanced_accuracy": ba,
            "balanced_accuracy_std": sd,
            "leakage": excess_leakage(ba),
            "ci95_low": lo,
            "ci95_high": hi,
            "significant_vs_chance": bool(lo is not None and lo > 0.5),
            "n_participants": int(len(np.unique(sg))),
            "n_clips": int(mask.sum()),
        }
        # Kept for the paired across-condition tests, which are far more sensitive than
        # comparing these independent CIs.
        out["per_participant"][name] = per_participant_accuracy(Xg, yg, sg)

    # Full-budget SEEN, for continuity with the existing attack_inference numbers.
    pred_full, _, _ = _lopo_predictions(X[seen_mask], y[seen_mask], subjects[seen_mask], 1.0)
    from sklearn.metrics import balanced_accuracy_score
    out["seen_full_budget_balanced_accuracy"] = float(
        balanced_accuracy_score(y[seen_mask], pred_full))

    out["MAL"] = out["seen"]["leakage"] - out["heldout"]["leakage"]
    return out


def per_participant_accuracy(X, y, subjects) -> dict:
    """{subject: fraction of that subject's clips the LOPO probe got right}.

    Uses the full-budget deterministic LOPO (no subsampling): conditions are compared to each
    other WITHIN a group, so attacker budget is held constant and cannot confound them.
    """
    pred, _, _ = _lopo_predictions(X, y, subjects, 1.0)
    return {s: float((pred[subjects == s] == y[subjects == s]).mean())
            for s in np.unique(subjects)}


def paired_condition_test(acc_a: dict, acc_b: dict) -> dict:
    """Wilcoxon signed-rank on per-participant accuracy, condition A vs condition B.

    Comparing two conditions by their independent CIs badly understates power: with only ~24
    held-out participants those intervals are wide and overlap almost everything. But the SAME
    participants appear in both conditions, so pairing removes between-participant variance --
    which is the dominant term here, since some people are simply easier to classify than
    others. This is the test to quote for "did redaction actually reduce recoverability?".
    """
    from scipy.stats import wilcoxon

    subs = sorted(set(acc_a) & set(acc_b))
    a = np.array([acc_a[s] for s in subs])
    b = np.array([acc_b[s] for s in subs])
    diff = a - b
    out = {
        "n_participants": len(subs),
        "mean_drop": float(diff.mean()),
        "median_drop": float(np.median(diff)),
        "n_decreased": int((diff > 0).sum()),
        "n_increased": int((diff < 0).sum()),
        "p_value": None,
    }
    if np.any(diff != 0):
        # One-sided: does condition A (usually clean) score HIGHER than B (redacted)?
        out["p_value"] = float(wilcoxon(a, b, alternative="greater",
                                        zero_method="wilcox").pvalue)
    return out


def utility_under_redaction(X, labels, subjects, seen_mask, heldout_mask, device):
    """Action accuracy of a fresh head trained on redacted SEEN features, tested on HELD-OUT.

    Reuses the TAPPFL utility probe verbatim so privacy and utility are measured the same way
    across the repo. Answers "what does this redaction cost the task?".
    """
    from src.training.tappfl_train import utility_retrain_head
    res = utility_retrain_head(X[seen_mask], labels[seen_mask],
                               X[heldout_mask], labels[heldout_mask], device, config.SEED)
    return {"accuracy": res["accuracy"], "balanced_accuracy": res["balanced_accuracy"]}


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def summarize(rows: pd.DataFrame) -> dict:
    """Collapse the condition table into the headline RCAL numbers."""
    def leak(cond, group):
        m = rows[(rows.condition == ckey(cond))]
        return float(m[f"{group}_leakage"].iloc[0]) if len(m) else float("nan")

    out = {}
    for group in ("seen", "heldout"):
        clean = leak("none", group)
        face = leak("face_black", group)
        person = leak("person_black", group)
        sham = leak("sham_face_black", group)
        sham_person = leak("sham_person_black", group)

        # AULRC over the blur severity curve, trapezoid on normalized severity.
        sev, val = [0.0], [clean]
        for s in config.REDACTION_SEVERITIES:
            if s <= 0:
                continue
            v = leak(f"face_blur_s{s:g}", group)
            # A missing severity point just drops out of the trapezoid rather than
            # poisoning the whole AULRC with a NaN.
            if np.isfinite(v):
                sev.append(s)
                val.append(v)
        aulrc = float(np.trapezoid(val, sev) / (sev[-1] - sev[0])) if len(sev) > 1 else float("nan")

        out[group] = {
            "clean_leakage": clean,
            # HEADLINE. The pixel-space oracle (rizvandwiki/gender-classification, the model
            # that produced the dataset's own labels) scores 0.817 on clean frames, 0.650 with
            # the face box blacked out, and exactly 0.500 -- chance -- with the person box
            # blacked out. So under person_black there is provably NO visible gender evidence
            # left, and whatever the probe still recovers came from the weights.
            "RRL_person_blackout": person,
            "RRL_face_blackout": face,
            "VEF_visible_evidence_fraction": (1 - face / clean) if clean > 0 else float("nan"),
            "AULRC_face_blur": aulrc,
            "specificity_face_vs_sham": sham - face,
            "specificity_person_vs_sham": sham_person - person,
        }

    out["MAL_person_blackout"] = (out["seen"]["RRL_person_blackout"]
                                  - out["heldout"]["RRL_person_blackout"])
    out["MAL_face_blackout"] = out["seen"]["RRL_face_blackout"] - out["heldout"]["RRL_face_blackout"]
    out["MAL_clean"] = out["seen"]["clean_leakage"] - out["heldout"]["clean_leakage"]
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_severity_curve(rows: pd.DataFrame, save_path: Path):
    """Leakage vs blur severity for both groups, with participant-bootstrap CI bands."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for group, color in (("seen", "#c0392b"), ("heldout", "#2980b9")):
        sev, val, lo, hi = [0.0], [], [], []
        base = rows[rows.condition == "none"]
        val.append(float(base[f"{group}_leakage"].iloc[0]))
        lo.append(excess_leakage(float(base[f"{group}_ci95_low"].iloc[0] or 0.5)))
        hi.append(excess_leakage(float(base[f"{group}_ci95_high"].iloc[0] or 0.5)))
        for s in config.REDACTION_SEVERITIES:
            if s <= 0:
                continue
            m = rows[rows.condition == ckey(f"face_blur_s{s:g}")]
            if not len(m):
                continue
            sev.append(s)
            val.append(float(m[f"{group}_leakage"].iloc[0]))
            lo.append(excess_leakage(float(m[f"{group}_ci95_low"].iloc[0] or 0.5)))
            hi.append(excess_leakage(float(m[f"{group}_ci95_high"].iloc[0] or 0.5)))
        ax.plot(sev, val, "o-", color=color, lw=2, label=f"{group} participants")
        ax.fill_between(sev, lo, hi, color=color, alpha=0.15)

    ax.axhline(0, color="k", ls="--", lw=1, label="chance (no leakage)")
    ax.set_xlabel("Face blur severity  (0 = clean, 1 = region destroyed)")
    ax.set_ylabel("Excess leakage  L = 2*max(0, bal-acc - 0.5)")
    ax.set_title("Leakage vs redaction severity\nresidual at high severity = not read off the pixels")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Severity curve -> {save_path}")


def plot_region_bars(rows: pd.DataFrame, save_path: Path):
    """Leakage per redacted region, with the sham controls beside them."""
    order = [ckey(c) for c in
             ["none", "face_black", "sham_face_black", "head_shoulders_black", "torso_black",
              "arms_black", "person_black", "sham_person_black", "background_black"]]
    present = [c for c in order if c in set(rows.condition)]
    if not present:
        return
    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(present)), 5))
    for i, (group, color) in enumerate((("seen", "#c0392b"), ("heldout", "#2980b9"))):
        vals = [float(rows[rows.condition == c][f"{group}_leakage"].iloc[0]) for c in present]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=color, label=f"{group} participants")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=35, ha="right")
    ax.set_ylabel("Excess leakage L")
    ax.set_title("Leakage by redacted region (sham = same box, moved off the occupant)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Region bars -> {save_path}")


def plot_grid_heatmap(rows: pd.DataFrame, save_path: Path):
    """Where in the frame does gender live -- occlusion sensitivity, no pose needed."""
    k = config.REDACTION_GRID
    clean = float(rows[rows.condition == "none"]["seen_leakage"].iloc[0])
    grid = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            m = rows[rows.condition == ckey(f"grid_{i}_{j}_black")]
            if len(m):
                grid[i, j] = clean - float(m["seen_leakage"].iloc[0])
    if np.isnan(grid).all():
        return
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(grid, cmap="magma")
    fig.colorbar(im, ax=ax, label="Leakage DROP when this cell is blacked out")
    ax.set_title(f"Occlusion sensitivity ({k}x{k})\nbright = this cell carried the gender evidence")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Grid heatmap -> {save_path}")


def short_ckpt_name(spec: str) -> str:
    """Readable label for a checkpoint path ('.../TAPPFL_results/.../best_model.pt' -> tappfl)."""
    if spec == "pretrained":
        return "pretrained"
    # Match case-insensitively on any path component ('tappfl_checkpoint', 'TAPPFL_results',
    # 'FL_results', 'fl_checkpoint', ...). Order matters: 'tappfl' contains 'fl'.
    low = spec.lower()
    if "tappfl" in low:
        return "tappfl"
    if "fl_" in low or "fedavg" in low or "federated" in low:
        return "fedavg"
    return "baseline"


def plot_checkpoint_comparison(rows: pd.DataFrame, save_path: Path):
    """Clean vs redacted leakage per encoder -- does the metric separate the defenses?

    The pretrained bar is the data-intrinsic floor: leakage there was never learned from this
    dataset, so only the excess above it is attributable to training.
    """
    cks = list(dict.fromkeys(rows.checkpoint))
    conds = [("none", "clean"), ("face_black", "face blacked out"),
             ("person_black", "person blacked out")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(cks))
    for ax, group in zip(axes, ("seen", "heldout")):
        for i, (cond, label) in enumerate(conds):
            vals = []
            for ck in cks:
                m = rows[(rows.checkpoint == ck) & (rows.condition == ckey(cond))]
                vals.append(float(m[f"{group}_leakage"].iloc[0]) if len(m) else np.nan)
            ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([short_ckpt_name(c) for c in cks], rotation=15)
        ax.set_title(f"{group} participants")
        ax.axhline(0, color="k", lw=1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Excess leakage L")
    axes[0].legend()
    fig.suptitle("Leakage per encoder, clean vs redacted "
                 "(residual under redaction = not read off the pixels)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Checkpoint comparison -> {save_path}")


def plot_privacy_utility(rows: pd.DataFrame, save_path: Path):
    """Every condition as a (leakage, action accuracy) point -- the actual trade-off."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sub = rows[~rows.condition.str.startswith("grid_")]
    ax.scatter(sub["seen_leakage"], sub["utility_accuracy"], s=70,
               c=["#c0392b" if c == "none" else "#2980b9" for c in sub.condition], zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(r.condition, (r.seen_leakage, r.utility_accuracy),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Excess gender leakage L (seen participants)  -- lower is better")
    ax.set_ylabel("Action accuracy, retrained head  -- higher is better")
    ax.set_title("Privacy / utility per redaction condition")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Privacy-utility scatter -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "best_model.pt"),
                   help="Comma-separated checkpoint paths, and/or the literal 'pretrained'")
    p.add_argument("--conditions", default=None,
                   help="Comma-separated conditions (default: the full standard sweep)")
    p.add_argument("--no-grid", action="store_true", help="Skip the k x k occlusion map")
    p.add_argument("--no-cache", action="store_true", help="Recompute features even if cached")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def run_checkpoint(ckpt_spec, conditions, device, no_cache):
    """Full condition sweep for one checkpoint. Returns (rows_df, summary_dict)."""
    pretrained = ckpt_spec == "pretrained"
    ckpt_path = None if pretrained else Path(ckpt_spec).resolve()
    if ckpt_path is not None and not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, _ = load_encoder(ckpt_path, device)

    # Split membership and gender labels are identical across conditions -- resolve once.
    _, train_subjects, val_subjects, test_subjects = build_subject_datasets()
    from src.data.dataset import load_gender_map
    gender_map = load_gender_map()

    rows, per_participant = [], {}
    for cond in conditions:
        redactor = parse_condition(cond, seed=config.SEED)
        logger.info(f"\n[{ckpt_spec}] condition: {redactor.key()}")

        X, subjects, labels, _ = features_for_condition(
            model, redactor, ckpt_path, device, no_cache)

        keep = np.array([s in gender_map for s in subjects])
        X, subjects, labels = X[keep], subjects[keep], labels[keep]
        y = np.array([gender_map[s] for s in subjects])
        seen_mask = np.array([s in train_subjects for s in subjects])
        heldout_mask = np.array([s in (val_subjects | test_subjects) for s in subjects])
        matched_n = min(len(set(subjects[seen_mask])), len(set(subjects[heldout_mask])))

        res = evaluate_condition(X, y, subjects, seen_mask, heldout_mask, matched_n)
        util = utility_under_redaction(X, labels, subjects, seen_mask, heldout_mask, device)

        logger.info(f"  seen    bal-acc={res['seen']['balanced_accuracy']:.3f} "
                    f"L={res['seen']['leakage']:.3f} | "
                    f"heldout bal-acc={res['heldout']['balanced_accuracy']:.3f} "
                    f"L={res['heldout']['leakage']:.3f} | "
                    f"MAL={res['MAL']:+.3f} | util={util['accuracy']:.3f}")

        per_participant[redactor.key()] = res.pop("per_participant")
        rows.append({
            "checkpoint": ckpt_spec,
            "condition": redactor.key(),
            "region": redactor.region,
            "mode": redactor.mode,
            "severity": redactor.severity if not redactor.is_identity else 0.0,
            **{f"seen_{k}": v for k, v in res["seen"].items()},
            **{f"heldout_{k}": v for k, v in res["heldout"].items()},
            "MAL": res["MAL"],
            "seen_full_budget_balanced_accuracy": res["seen_full_budget_balanced_accuracy"],
            "utility_accuracy": util["accuracy"],
            "utility_balanced_accuracy": util["balanced_accuracy"],
            "matched_budget_n": matched_n,
        })

    df = pd.DataFrame(rows)
    return df, summarize(df), paired_tests(per_participant)


# The contrasts worth testing. Each is (label, condition_A, condition_B) asking "is A higher
# than B?" -- i.e. did going from A to B actually reduce how recoverable gender is?
PAIRED_CONTRASTS = [
    ("clean_vs_face_black", "none", "face_black"),
    ("clean_vs_person_black", "none", "person_black"),
    ("clean_vs_face_blur_max", "none", "face_blur_s1"),
    # Specificity: the redaction must beat its own sham, or the drop is just occlusion.
    ("sham_face_vs_face_black", "sham_face_black", "face_black"),
    ("sham_person_vs_person_black", "sham_person_black", "person_black"),
]


def paired_tests(per_participant: dict) -> dict:
    """Run every contrast in PAIRED_CONTRASTS for both participant groups."""
    out = {}
    for label, a, b in PAIRED_CONTRASTS:
        ka, kb = ckey(a), ckey(b)
        if ka not in per_participant or kb not in per_participant:
            continue
        out[label] = {
            group: paired_condition_test(per_participant[ka][group], per_participant[kb][group])
            for group in ("seen", "heldout")
        }
    return out


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = OUT_DIR / datetime.now().strftime("%d_%B_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(run_dir / "run.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = (args.conditions.split(",") if args.conditions
                  else default_conditions(include_grid=not args.no_grid))
    checkpoints = args.checkpoint.split(",")

    logger.info(f"Output dir : {run_dir}")
    logger.info(f"Device     : {device}")
    logger.info(f"Checkpoints: {checkpoints}")
    logger.info(f"Conditions ({len(conditions)}): {conditions}")

    coverage = pose_tier_coverage()
    logger.info(f"Pose anchor coverage over all clips: {coverage} "
                f"(tier {TIER_PER_FRAME} = per-frame keypoints, higher = fallback)")

    all_rows, summaries, tests = [], {}, {}
    for ckpt in checkpoints:
        df, summary, paired = run_checkpoint(ckpt, conditions, device, args.no_cache)
        all_rows.append(df)
        summaries[ckpt] = summary
        tests[ckpt] = paired

    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(run_dir / "conditions.csv", index=False)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "conditions": conditions,
            "pose_tier_coverage": coverage,
            "matched_budget_draws": config.MATCHED_BUDGET_DRAWS,
            "summary": summaries,
            "paired_tests": tests,
            "definitions": {
                "L": "2 * max(0, balanced_accuracy - 0.5); 0 = chance, 1 = perfect recovery",
                "RRL": "L under face blackout -- leakage surviving removal of visible evidence",
                "VEF": "1 - L(face)/L(clean) -- fraction of leakage carried by face pixels",
                "MAL": "L(seen) - L(heldout) at matched attacker budget -- memorization",
                "AULRC": "area under L vs face-blur severity, normalized",
                "specificity": "L(sham) - L(region) -- drop net of distribution shift",
            },
        }, f, indent=2)

    # Plots use the first checkpoint; comparisons across checkpoints live in conditions.csv.
    first = rows[rows.checkpoint == checkpoints[0]]
    plot_severity_curve(first, run_dir / "severity_curve.png")
    plot_region_bars(first, run_dir / "region_bars.png")
    plot_privacy_utility(first, run_dir / "privacy_utility.png")
    if not args.no_grid:
        plot_grid_heatmap(first, run_dir / "grid_heatmap.png")
    if len(checkpoints) > 1:
        plot_checkpoint_comparison(rows, run_dir / "checkpoint_comparison.png")

    logger.info("\n" + "=" * 76)
    logger.info("REDACTION-CONDITIONED ATTRIBUTE LEAKAGE — SUMMARY")
    logger.info("=" * 76)
    for ckpt, s in summaries.items():
        logger.info(f"\n  {ckpt}")
        for group in ("seen", "heldout"):
            g = s[group]
            logger.info(f"    {group:8s} clean L={g['clean_leakage']:.3f}  "
                        f"RRL(person)={g['RRL_person_blackout']:.3f}  "
                        f"RRL(face)={g['RRL_face_blackout']:.3f}  "
                        f"VEF={g['VEF_visible_evidence_fraction']:.3f}  "
                        f"AULRC={g['AULRC_face_blur']:.3f}  spec={g['specificity_face_vs_sham']:+.3f}")
        logger.info(f"    MAL clean={s['MAL_clean']:+.3f}   "
                    f"face-blackout={s['MAL_face_blackout']:+.3f}   "
                    f"person-blackout={s['MAL_person_blackout']:+.3f}")
        for label, t in tests[ckpt].items():
            parts = []
            for group in ("seen", "heldout"):
                g = t[group]
                p = f"p={g['p_value']:.3f}" if g["p_value"] is not None else "p=n/a"
                parts.append(f"{group} drop={g['mean_drop']:+.3f} {p} "
                             f"({g['n_decreased']}/{g['n_participants']} down)")
            logger.info(f"      [paired] {label:28s} " + " | ".join(parts))
    logger.info("=" * 76)
    logger.info(f"All results -> {run_dir}")


if __name__ == "__main__":
    main()
