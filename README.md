# Naturalistic Driving — Action Recognition & Privacy-Preserving Federated Learning

Video action recognition on in-car naturalistic driving footage, with a federated-learning
setup and a **TAPPFL** privacy defense that stops the learned representation from leaking
participant sensitive attribure e.g.  **gender**.

- **Task:** classify 8 in-car actions from short video clips.
- **Model:** VideoMAE (`MCG-NJU/videomae-base`) — 16 frames @ 224×224 → 768-d clip
  representation → linear action head.
- **Data:** 77 participants, ~8 clips each (FrontWeb camera, "Action" clips).
- **Labels:** `Dance`, `InteractWithRadio`, `ScaryReaction`, `TalkToSelf`, `Tired`,
  `TouchPassengerSeat`, `WaveToLeft`, `WaveToRight`.
- **Sensitive attribute:** participant gender (binary), used only to *measure* and *defend
  against* attribute leakage.

Splits are **participant-level 70/15/15** (train/val/test), so no subject appears in two
splits.

---

## Repository layout

```
src/
  config.py                     # all paths + hyperparameters (single source of truth)
  data/dataset.py               # datasets, participant splits, frame cache, gender labels
  data/pose.py                  # OpenPose keypoints -> face/torso/arm/person region boxes
  data/redaction.py             # blur / black-box / pixelate occlusion operators + controls
  preprocessing/extract_genders.py   # derive participant gender labels from video frames
  training/
    train.py                    # centralized baseline training
    federated_train.py          # federated baseline (FedAvg, 1 client = 1 participant)
    tappfl_train.py             # TAPPFL: privacy-preserving federated training
  evaluation/
    evaluate.py                 # task metrics for a checkpoint
    attack_inference.py         # gender attribute-inference attack (MLP probe)
    redaction_leakage.py        # RCAL: leakage vs redaction, separating memorization
data/                           # raw dataset: CSVs + Video1..4 (NOT in git; folder or symlink)
annot/                          # reviewed participant gender/race labels (tracked)
artifacts/                      # outputs, frame_cache, wandb runs (git-ignored)
notebooks/                      # EDA
```

---

## Setup

```bash
pip install -r requirements.txt        # torch/torchvision already installed with CUDA
cp .env.example .env                   # then add your WANDB_API_KEY (or use --no-wandb)
```

All scripts are **modules** and must be run from the repo root:

```bash
python -m src.training.tappfl_train --help
```

Clips are decoded once into `artifacts/frame_cache/` and reused, so the first run is slower.

---

## Usage

```bash
# Centralized baseline  (no CLI args — configure via src/config.py)
python -m src.training.train

# Federated baseline (FedAvg; each participant is a client)
python -m src.training.federated_train --rounds 30

# TAPPFL — privacy-preserving federated training
python -m src.training.tappfl_train --rounds 30 --local-epochs 3 --lambda 0.5

# Evaluate a checkpoint on the task
python -m src.evaluation.evaluate --checkpoint <path/to/best_model.pt>

# Gender attribute-inference attack on a checkpoint
python -m src.evaluation.attack_inference --checkpoint <path/to/best_model.pt>

# Redaction-conditioned leakage (RCAL) — see "How results are measured"
python -m src.evaluation.redaction_leakage --checkpoint <path/to/best_model.pt>
```

Each run writes a timestamped folder under `artifacts/outputs/` containing `checkpoints/`,
`metrics.json`, `round_history.json`, plots, and `train.log`.

---

## How TAPPFL works here

Three networks; **only the encoder is federated**:

| Network | Role | Communicated? |
| --- | --- | --- |
| `f_Θ` — VideoMAE encoder | produces the representation `z` | **yes** (FedAvg) |
| `g_Ψ` — privacy adversary (MLP) | predicts gender from `z` | no |
| `h_Ω` — JSD MI critic | estimates `I(x; z \| u)` | no |

Per batch (mirrors the original TAPPFL 1:1:1 minimax):

```
total_loss = -λ · CE(g_Ψ(z), gender) + (1-λ) · (-jsd_MI(h_Ω, x, z, gender))
```
one encoder step, one adversary step, one MI-critic step. **λ weights privacy**
(λ=1 → pure privacy, λ=0 → pure utility). Aggregation is uniform FedAvg.

Two adaptations to this dataset (documented in each run's `tappfl_config.json`):

1. **Gender-balanced replay buffer.** Each client is one participant, so each client is
   single-gender and a per-client adversary would forget across clients. A shared buffer of
   recent `(z, gender)` lets `g_Ψ` train on both genders every step.
2. **Adversary relaunch.** Fooling one fixed adversary does *not* remove the attribute. Each
   round `g_Ψ` is re-initialized and re-trained to competence on the current representation
   **before** the encoder trains against it, so the encoder must beat a *fresh* attacker.

---

## How results are measured

Both metrics **freeze the encoder** and train a fresh probe on the 768-d `z`:

- **Utility** — train a new MLP head (`768→256→8`) on the *training* subjects' `z`, report
  action accuracy on the *held-out test* subjects. (Measures task info retained in `z`, not
  the fit of a stale head.)
- **Privacy leakage** (`attack_inference.py`) — train a logistic gender probe on frozen `z`
  under **leave-one-participant-out**: for each person, train on all the others and predict
  theirs. Report **balanced accuracy** with a 95% CI bootstrapped over *participants*.

**Matched attacker budget.** SEEN has 53 participants and HELD-OUT 24, so a plain LOPO
comparison hands the SEEN attacker 52 training participants against HELD-OUT's 23. Attacker
budget dominates this measurement (11 → .574, 26 → .647, 52 → .741 on the same encoder), so the
raw pair is **not comparable** — which is why HELD-OUT used to score *above* SEEN. Both groups
are now also probed with the attacker subsampled to the smaller group's size, averaged over
`MATCHED_BUDGET_DRAWS` draws. Quote the matched number for any seen-vs-held-out claim.

---

## Redaction-Conditioned Attribute Leakage (RCAL)

A single leakage number cannot tell apart two very different privacy stories:

1. **Input-visible leakage** — gender is readable off the pixels (face, hair, build, clothing).
   Any encoder, even an untrained one, shows this. *The data leaks, not the model.*
2. **Memorized leakage** — the encoder baked participant-specific information into its weights
   and recovers gender even when the pixel evidence is gone. *The model leaks* — and this is
   the part that makes releasing weights risky.

`src/evaluation/redaction_leakage.py` separates them by **redacting a region of every clip
before attribute inference** and watching how leakage decays. Models are *not* retrained: the
clean-trained checkpoint is frozen, redacted clips go through it, and the probe is retrained on
the resulting features, so the attacker always adapts to the redaction.

Regions come from the OpenPose-25B keypoints in `data/Pose_OpenPose25b/` (`src/data/pose.py`);
the occlusion operators — blur, black-box, pixelation, plus a `k×k` grid map — live in
`src/data/redaction.py`.

With `L = 2·max(0, balanced_accuracy − 0.5)`:

| Metric | Definition | Reads as |
| --- | --- | --- |
| **RRL** | `L(person-blackout)` | Leakage surviving removal of all visible evidence |
| **VEF** | `1 − L(face)/L(clean)` | Fraction of leakage carried by face pixels |
| **MAL** | `L(seen) − L(held-out)`, matched budget | Memorization-attributable leakage |
| **AULRC** | `∫ L(face-blur@s) ds` | One scalar per model, for comparing defenses |
| **Specificity** | `L(sham) − L(region)` | Region-specific drop, net of distribution shift |

Three controls carry the argument:

- **Sham** — an identically sized box placed *away* from the occupant. The encoder was trained
  on clean video, so any occlusion is distribution shift; only the drop **beyond sham** is
  attributable to the region's content.
- **Pretrained floor** — `--checkpoint pretrained` runs the sweep on the un-fine-tuned encoder.
  Leakage above that floor is what fine-tuning on this data added.
- **Pixel oracle** — the same classifier that produced the gender labels
  (`extract_genders.py`) scores **0.817** on clean frames, **0.650** under `face_black`, and
  exactly **0.500 — chance** — under `person_black`, while `sham_face_black` stays at **0.800**.
  So under `person_black` no visible gender evidence remains, and the redaction is verified to
  be region-specific rather than an occlusion artifact.

```bash
# smoke test
python -m src.evaluation.redaction_leakage --conditions none,face_black,sham_face_black --no-grid
# full sweep, comparing defenses against the data-intrinsic floor
python -m src.evaluation.redaction_leakage --checkpoint pretrained,<baseline>,<fedavg>,<tappfl>
```

### Reading the results — use the paired tests, not the CIs

With only 24 held-out participants the per-condition bootstrap CIs are ~±0.14 and overlap almost
everything, so comparing two conditions by their independent CIs badly understates power. The
same participants appear in both conditions, so the driver also runs a **Wilcoxon signed-rank
test on per-participant probe accuracy**, which removes between-participant variance (the
dominant term — some people are simply easier to classify). Quote that.

The contrast that carries the argument is `sham_face_black` vs `face_black`: does covering the
*face* beat covering an *equally sized patch of upholstery*? Measured across four independently
trained encoders:

| Encoder | SEEN drop | p | HELD-OUT drop | p |
| --- | --- | --- | --- | --- |
| pretrained | +0.027 | 0.172 | +0.160 | **0.010** |
| baseline | −0.008 | 0.581 | +0.083 | 0.066 |
| fedavg | +0.022 | 0.236 | +0.094 | **0.023** |
| tappfl | +0.029 | 0.190 | +0.128 | **0.020** |

Removing the face significantly reduces gender recoverability for participants the model never
saw, but produces **no significant reduction for participants it trained on** (p = 0.17–0.58 in
every case). For training subjects the attribute is recoverable without the face being visible
at all — the signature of memorization, replicating across all four encoders.

**Blurring is a weak redaction.** Even at maximum severity, face-blur never significantly
reduces held-out leakage (p = 0.47–0.98) and sometimes *raises* it, because it preserves hair
colour, skin tone and clothing. Only hard occlusion works. Redacting the background instead of
the person also raises measured leakage, by removing scene clutter that the probe was partly
keying on.

**Caveats.**
- FrontWeb pose stops near frame ~3270 while clips run to ~4600, so only ~63% of clips have
  per-frame keypoints; the rest anchor on the subject's median box (the camera is fixed and
  subjects are belted). Every run logs the per-tier fractions.
- Absolute per-condition values are noisy at this participant count. The face-blur severity
  curve is non-monotonic, and individual outliers occur (baseline `person_black` on held-out
  reads 0.659 against 0.507–0.569 for the other three encoders). Treat single cells as
  indicative; trust the paired tests and the replication across encoders.




