from pathlib import Path

# Repo root: this file lives at <root>/src/config.py, so parents[1] is the root.
BASE_DIR = Path(__file__).resolve().parents[1]

# Input data (raw videos, master table) -- not in git.
DATA_DIR = BASE_DIR / "data"
# Reviewed participant label CSVs -- tracked in git.
ANNOT_DIR = BASE_DIR / "annot"
# Generated artifacts (run outputs, checkpoints, decoded-frame cache, wandb logs).
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CSV_PATH = DATA_DIR / "DEFADSMasterTableV1.csv"

VIDEO_ROOTS = {
    1: DATA_DIR / "Video1" / "Video1",
    2: DATA_DIR / "Video2" / "Video2",
    3: DATA_DIR / "Video3" / "Video3",
    4: DATA_DIR / "Video4" / "Video4",
}

# Per-frame OpenPose-25B keypoints, one CSV per {subject}_{camera}_{videotype}, in the
# ORIGINAL video coordinate frame (1920x1080 for FrontWeb/RVWeb). Used to localize the
# face / body regions that the redaction-based leakage metric occludes.
POSE_ROOT = DATA_DIR / "Pose_OpenPose25b" / "Pose_OpenPose25b"

OUTPUT_DIR = ARTIFACTS_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# wandb writes runs to <WANDB_DIR>/wandb/; the trainers export this before importing wandb.
WANDB_DIR = ARTIFACTS_DIR

# Decoded-frame cache: each clip's sampled frames are decoded once and reused across epochs/rounds.
# Frames are downscaled to a CACHE_SHORT_EDGE short edge before caching (the image processor
# resizes to a 224 short edge + center-crops anyway), keeping the cache small (~3 GB vs ~60 GB raw).
FRAME_CACHE_DIR = ARTIFACTS_DIR / "frame_cache"
USE_FRAME_CACHE = True
CACHE_SHORT_EDGE = 256
CACHE_WARM_WORKERS = 32   # parallel processes used to pre-decode the cache

MODEL_NAME = "MCG-NJU/videomae-base"

LABEL_NAMES = (
    "Dance",
    "InteractWithRadio",
    "ScaryReaction",
    "TalkToSelf",
    "Tired",
    "TouchPassengerSeat",
    "WaveToLeft",
    "WaveToRight",
)
NUM_CLASSES = len(LABEL_NAMES)

NUM_FRAMES = 16
IMAGE_SIZE = 224

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

BATCH_SIZE = 8
NUM_WORKERS = 8
NUM_EPOCHS = 60

LR_BACKBONE = 5e-6
LR_HEAD = 2e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
GRAD_CLIP = 1.0
AMP = True

# Data filtering (parameterized so other cameras / video types can be used later)
DATA_VIDEO_TYPE = "Action"
DATA_CAMERA = "FrontWeb"

# Federated learning
FL_ROUNDS = 30
FL_LOCAL_EPOCHS = 2
FL_CLIENT_FRACTION = 1.0           # 1.0 = full participation (all clients each round)
FL_CLIENT_BATCH_SIZE = BATCH_SIZE
FED_OUTPUT_DIR = OUTPUT_DIR / "FL_results"   # each run gets its own timestamped subfolder


GENDER_CSV = ANNOT_DIR / "participant_genders_manual.csv"   # Subject -> {male, female}

# Participant race, produced by preprocessing/extract_races.py (VLM + face-attribute classifier
# polled over 7 images each). These are PERCEIVED-race annotations inferred from appearance, not
# self-reported demographics -- usable for measuring attribute leakage, not for reporting cohort
# demographics. The auto output (participant_races.csv) has been reviewed; the file below is
# that review's result and is what downstream code should read. Its Source column marks which
# rows a human overrode -- currently P051 only, latino -> white.
RACE_CSV = ANNOT_DIR / "participant_races_manual.csv"       # Subject -> RACE_CLASSES
RACE_CLASSES = ("white", "black", "latino", "asian")
TAPPFL_OUTPUT_DIR = OUTPUT_DIR / "TAPPFL_results"          # one timestamped subfolder per run

# Follows the original TAPPFL local minimax (github.com/TAPPFL/TAPPFL): per batch, one
# feature-extractor update on total_loss = -lambda*L_adv + (1-lambda)*L_JSD, one adversary
# update (CE on the true private label), and one MI-estimator update -- no warm-up and no
# repeated adversary steps. lambda WEIGHTS THE PRIVACY (adversarial) TERM: lambda=1 => pure
# privacy (original default), lambda=0 => pure utility. FedAvg is uniform.
TAPPFL_K = 2                 # number of sensitive-attribute classes (binary gender)
TAPPFL_ROUNDS = 30           # federated rounds (R)
TAPPFL_LOCAL_EPOCHS = 5      # local epochs per client per round (original uses local_ep=10)
TRADEOFF_LAMBDA = 0.5        # lambda: weight on the privacy term (1=pure privacy, 0=pure utility)
# Original trains fe/adv/mi all at lr=0.01 from scratch. Here the FE is a *fine-tuned*
# VideoMAE, so it keeps a small LR; the from-scratch adversary/MI use a higher LR.
TAPPFL_LR_INIT = 5e-5        # FE (backbone) learning rate
TAPPFL_ADV_LR = 1e-3         # adversary g_Psi and MI estimator h_Omega learning rate
TAPPFL_CLIP = 10.0           # gradient clipping L2 norm for all networks (stability guard)
TAPPFL_X_POOL = 16           # spatial side S of the pooled-pixel x surrogate (3*S*S = x dim)
# Cross-client gender-balanced replay buffer for the shared adversary g_Psi. Because each
# client is single-gender, training g_Psi on the current client alone makes it forget across
# clients (adv stuck at chance). Sampling a class-balanced batch from a buffer of recent z
# keeps g_Psi a competent gender classifier, so the FE gets a coherent "remove gender" signal.
TAPPFL_BUFFER_CAP = 256      # per-class capacity of the (z, gender) replay buffer
# Adversary relaunch: fooling one fixed g_Psi does NOT remove the attribute (a fresh probe
# still recovers it). So every N rounds we re-initialize g_Psi and re-train it to competence
# on the buffer BEFORE the encoder trains against it -> the encoder must beat a fresh attacker
# each round, aligning training with the fresh-probe evaluation.
TAPPFL_ADV_RESET_EVERY = 1      # re-initialize + re-warm g_Psi every N rounds (0 = never)
TAPPFL_ADV_WARMUP_STEPS = 300   # steps to train a freshly relaunched g_Psi on the buffer
# Utility is measured by retraining a fresh head on the FROZEN encoder (centrally, on the
# training subjects) -> test action accuracy. This measures task info retained in z, not the
# fit of a stale head. Privacy (gender leakage) is measured on the TRAINING (seen) clients.
TAPPFL_UTIL_HEAD_EPOCHS = 150   # epochs to train the fresh MLP utility head on frozen z


# ---------------------------------------------------------------------------
# Redaction-Conditioned Attribute Leakage (RCAL)
# ---------------------------------------------------------------------------
# A plain leakage number conflates "gender is visible in the pixels" (any encoder would show
# it) with "the encoder memorized the attribute into its weights" (the actual model-release
# risk). Redacting the informative regions before attribute inference separates the two:
# leakage that SURVIVES redaction is not being read off the picture.

# The *_OPo25B.csv files are OpenPose BODY_25B (COCO joint order) but their column HEADERS use
# BODY_25 names, so e.g. the "RightShoulder" column is really the right eye and "RightWrist"
# is the right ear. src/data/pose.py renames columns by position to the true BODY_25B joints;
# these names refer to those. Measured over 20 subjects, fraction of frames with confidence >
# POSE_CONF_MIN: Nose .99, LEye .99, REye .96, LEar .95, REar .77, RShoulder .81,
# LShoulder .73, UpperNeck .37, HeadTop .22. Elbows and wrists sit at <=.01 (the arms are
# below the dash / out of frame), so arm regions cannot be localized from pose.
POSE_KEYPOINTS = (
    "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder",
    "LElbow", "RElbow", "LWrist", "RWrist", "UpperNeck", "HeadTop",
)
POSE_CONF_MIN = 0.3

# Box sizes are fractions of the FRAME, not of a pose-derived body scale. Shoulder width was
# tried first and is unusable: when the subject leans, one shoulder is occluded and the span
# collapses, so the derived scale swings ~6x within a single clip (P001: 232 -> 635 px) and
# ~4x between subjects on a fixed camera rig. The camera is fixed and every subject sits in
# the driver's seat, so apparent head size is near-constant and a frame-relative box is far
# more stable. Values below are ~1.35x the measured head extent, calibrated visually.
FACE_HALF_W = 0.080         # half-width  as a fraction of frame WIDTH
FACE_HALF_H = 0.240         # half-height as a fraction of frame HEIGHT
FACE_CENTER_DY = -0.05      # the Head keypoint is the nose; face center sits above it, and
                            # hair (a strong gender cue) extends further up still
TORSO_HALF_W = 0.16         # torso half-width, fraction of frame width
ARM_HALF = 0.075            # arm/hand box half-size, fraction of frame width

# Clips with no pose rows at all fall back to the subject's whole-video median box, dilated to
# absorb head motion during the high-movement actions (Dance, WaveToLeft).
FALLBACK_DILATE = 1.25

# The pose files cover every video frame (row count == video frame count), so nearly all
# clips get per-frame anchors; frames with a missing/low-confidence nose resolve through the
# median-anchor fallback. The driver logs the per-tier fractions.

REDACTION_SEVERITIES = (0.0, 0.125, 0.25, 0.5, 1.0)
REDACTION_GRID = 4          # k x k occlusion map -> "where in the frame does gender live"

# SEEN (train-split) has 53 participants and HELD-OUT (val+test) has 24, so a plain LOPO
# comparison gives the SEEN attacker 52 training participants against the HELD-OUT attacker's
# 23. Attacker budget dominates this measurement (11 -> .574, 26 -> .647, 52 -> .741 balanced
# accuracy on the same undefended encoder), so the groups must be compared at MATCHED budget:
# subsample SEEN participants down to the HELD-OUT count, repeat, average.
MATCHED_BUDGET_DRAWS = 20

REDACTION_OUTPUT_DIR = OUTPUT_DIR / "attack" / "redaction"
