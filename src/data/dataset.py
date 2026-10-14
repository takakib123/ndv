import os
import random
import warnings
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import AutoImageProcessor
from tqdm import tqdm

from src.config import (
    CSV_PATH, VIDEO_ROOTS, MODEL_NAME,
    LABEL_NAMES, NUM_FRAMES, SEED,
    DATA_VIDEO_TYPE, DATA_CAMERA,
    FRAME_CACHE_DIR, USE_FRAME_CACHE, CACHE_SHORT_EDGE,
    GENDER_CSV,
)


def get_video_path(subject: str, camera: str = DATA_CAMERA, video_type: str = DATA_VIDEO_TYPE):
    n = int(subject[1:])
    if n <= 26:
        folder = 1
    elif n <= 51:
        folder = 2
    elif n <= 75:
        folder = 3
    else:
        folder = 4
    path = VIDEO_ROOTS[folder] / f"{subject}_{camera}_{video_type}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def sample_frames(start: int, end: int, n: int = NUM_FRAMES) -> list:
    if end - start >= n:
        return np.linspace(start, end - 1, n).astype(int).tolist()
    # Safety fallback: tile short clips (does not occur in this dataset)
    indices = list(range(start, end))
    tiled = (indices * math.ceil(n / len(indices)))[:n]
    return sorted(tiled)


def read_frames(video_path: str, frame_indices: list) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If seek fails, use the last successfully read frame or a blank
            frame = frames[-1] if frames else np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W, 3) uint8


def _resize_short_edge(frames: np.ndarray, short: int) -> np.ndarray:
    """Resize each (H, W, 3) frame so its shorter edge == `short`, preserving aspect ratio.
    Downscale only (no upscaling). frames: (T, H, W, 3) uint8."""
    _, h, w, _ = frames.shape
    if min(h, w) <= short:
        return frames
    if h < w:
        new_h, new_w = short, round(w * short / h)
    else:
        new_h, new_w = round(h * short / w), short
    return np.stack([cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames])


def _cache_path(video_path: str, start: int, end: int) -> Path:
    # Unique per clip: the filename stem encodes subject_camera_videotype, and start/end + NUM_FRAMES
    # fully determine the deterministically sampled frames.
    stem = Path(video_path).stem
    return FRAME_CACHE_DIR / f"{stem}_{start}_{end}_{NUM_FRAMES}.npy"


def read_frames_cached(video_path: str, start: int, end: int) -> np.ndarray:
    """Decode a clip's sampled frames, caching the raw (pre-augmentation) result to disk so each
    clip is decoded only once and reused across every epoch / federated round."""
    if not USE_FRAME_CACHE:
        return read_frames(video_path, sample_frames(start, end))

    cf = _cache_path(video_path, start, end)
    if cf.exists():
        return np.load(cf)

    frames = read_frames(video_path, sample_frames(start, end))
    frames = _resize_short_edge(frames, CACHE_SHORT_EDGE)
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic write so concurrent warmers / workers never observe a partial file.
    tmp = cf.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, frames)
    os.replace(tmp, cf)
    return frames


def _warm_one(args):
    video_path, start, end = args
    read_frames_cached(video_path, start, end)


def warm_frame_cache(records, workers: int = 16):
    """Pre-decode every unique clip in `records` in parallel so the first epoch is not I/O bound."""
    tasks = sorted({(r["video_path"], r["start_frame"], r["end_frame"]) for r in records})
    missing = [t for t in tasks if not _cache_path(*t).exists()]
    if not missing:
        print(f"Frame cache: all {len(tasks)} clips already cached.")
        return
    print(f"Frame cache: decoding {len(missing)}/{len(tasks)} clips with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(tqdm(ex.map(_warm_one, missing), total=len(missing), desc="Warming frame cache"))


@lru_cache(maxsize=4)
def get_image_processor(model_name: str = MODEL_NAME):
    """Memoized image processor.

    `from_pretrained` re-validates against the HF hub on every call, which costs seconds when
    the network is slow or unreachable. The RCAL sweep builds one dataset per redaction
    condition per checkpoint (100+ constructions), so this has to be resolved once.
    """
    return AutoImageProcessor.from_pretrained(model_name)


_AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
])


class ActionVideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: dict, augment: bool = False,
                 camera: str = DATA_CAMERA, video_type: str = DATA_VIDEO_TYPE,
                 redactor=None):
        # `redactor` must stay LAST: GenderActionVideoDataset and SubjectAwareVideoDataset
        # both forward to this constructor positionally.
        self.augment = augment
        self.redactor = redactor
        self.processor = get_image_processor()
        self.records = []

        for _, row in df.iterrows():
            subject = row["Subject"]
            try:
                video_path = str(get_video_path(subject, camera, video_type))
            except FileNotFoundError as e:
                warnings.warn(str(e))
                continue
            self.records.append({
                "video_path": video_path,
                "start_frame": int(row["StartFrame"]),
                "end_frame": int(row["EndFrame"]),
                "label_idx": label2idx[row["Label"]],
                "subject": subject,
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # (T, H, W, 3) uint8 — decoded once and cached; augmentation below still varies per access.
        frames_np = read_frames_cached(rec["video_path"], rec["start_frame"], rec["end_frame"])

        # Redaction (blur / black-box / pixelate over a pose-derived region) is applied to the
        # cached frames before the processor's resize+crop, so it needs no re-decoding and
        # leaves the frame cache valid. Used by the redaction-conditioned leakage metric.
        if self.redactor is not None and not self.redactor.is_identity:
            frames_np = self.redactor.apply(
                frames_np, rec["subject"],
                sample_frames(rec["start_frame"], rec["end_frame"]),
                rec["video_path"],
            )

        if self.augment:
            # Convert to (T, C, H, W) uint8 tensor for v2 transforms
            t = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)
            t = _AUGMENT(t)
            # Convert back to list of PIL images for processor
            pil_frames = [Image.fromarray(t[i].permute(1, 2, 0).numpy()) for i in range(t.shape[0])]
        else:
            pil_frames = [Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]

        # Processor expects a list of PIL images; returns pixel_values (1, T, C, H, W)
        encoding = self.processor(images=pil_frames, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)  # (T, C, H, W)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(rec["label_idx"], dtype=torch.long),
        }


def load_filtered_df(csv_path=CSV_PATH, video_type=DATA_VIDEO_TYPE, camera=DATA_CAMERA):
    """Read the master CSV and keep only the requested video type / camera rows."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df = df[(df["VideoType"] == video_type) & (df["Camera"] == camera)].copy()
    df = df.dropna(subset=["StartFrame", "EndFrame"])
    df["StartFrame"] = df["StartFrame"].astype(int)
    df["EndFrame"] = df["EndFrame"].astype(int)
    return df


def _split_subjects(df: pd.DataFrame, seed=SEED):
    """Participant-level 70/15/15 split. Returns (train, val, test) subject sets."""
    subjects = sorted(df["Subject"].unique())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_train = int(len(subjects) * 0.70)
    n_val = int(len(subjects) * 0.15)
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])
    return train_subjects, val_subjects, test_subjects


def build_datasets(csv_path=CSV_PATH, seed=SEED, video_type=DATA_VIDEO_TYPE, camera=DATA_CAMERA,
                   redactor=None):
    """`redactor` (src.data.redaction.Redactor) is applied to every split, e.g. to train and
    test a model on face-blurred video. None trains on clean video."""
    df = load_filtered_df(csv_path, video_type, camera)
    train_subjects, val_subjects, test_subjects = _split_subjects(df, seed)

    label2idx = {label: i for i, label in enumerate(LABEL_NAMES)}

    train_df = df[df["Subject"].isin(train_subjects)]
    val_df = df[df["Subject"].isin(val_subjects)]
    test_df = df[df["Subject"].isin(test_subjects)]

    print(f"Subjects — train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(test_subjects)}")
    print(f"Clips   — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_ds = ActionVideoDataset(train_df, label2idx, augment=True, camera=camera, video_type=video_type,
                                  redactor=redactor)
    val_ds = ActionVideoDataset(val_df, label2idx, augment=False, camera=camera, video_type=video_type,
                                  redactor=redactor)
    test_ds = ActionVideoDataset(test_df, label2idx, augment=False, camera=camera, video_type=video_type,
                                  redactor=redactor)

    return train_ds, val_ds, test_ds


def build_federated_datasets(csv_path=CSV_PATH, seed=SEED, video_type=DATA_VIDEO_TYPE, camera=DATA_CAMERA,
                             redactor=None):
    """Federated variant of build_datasets.

    Uses the same filtering + 70/15/15 participant split as build_datasets, but returns the
    training participants as separate per-client datasets (each client = one subject) instead of
    one pooled training set. Val/test remain centralized held-out sets. `redactor` is applied to
    every split, as in build_datasets.

    Returns: (client_datasets, val_ds, test_ds, label2idx)
      client_datasets: dict[subject_id -> ActionVideoDataset]  (augment=True)
    """
    df = load_filtered_df(csv_path, video_type, camera)
    train_subjects, val_subjects, test_subjects = _split_subjects(df, seed)

    label2idx = {label: i for i, label in enumerate(LABEL_NAMES)}

    client_datasets = {}
    for subject in sorted(train_subjects):
        sub_df = df[df["Subject"] == subject]
        client_datasets[subject] = ActionVideoDataset(
            sub_df, label2idx, augment=True, camera=camera, video_type=video_type, redactor=redactor
        )

    val_df = df[df["Subject"].isin(val_subjects)]
    test_df = df[df["Subject"].isin(test_subjects)]
    val_ds = ActionVideoDataset(val_df, label2idx, augment=False, camera=camera, video_type=video_type,
                                  redactor=redactor)
    test_ds = ActionVideoDataset(test_df, label2idx, augment=False, camera=camera, video_type=video_type,
                                  redactor=redactor)

    n_client_clips = sum(len(ds) for ds in client_datasets.values())
    print(f"Federated — clients: {len(client_datasets)}, val subjects: {len(val_subjects)}, "
          f"test subjects: {len(test_subjects)}")
    print(f"Clips   — clients(total): {n_client_clips}, val: {len(val_df)}, test: {len(test_df)}")

    return client_datasets, val_ds, test_ds, label2idx


# ---------------------------------------------------------------------------
# TAPPFL: gender-aware datasets (sensitive attribute u = participant gender)
# ---------------------------------------------------------------------------

def load_gender_map(gender_csv=GENDER_CSV) -> dict:
    """Map Subject -> gender index (0=female, 1=male) from the manual gender CSV."""
    gdf = pd.read_csv(gender_csv)
    gdf.columns = gdf.columns.str.strip()
    gdf = gdf.dropna(subset=["Subject", "Gender"])
    gdf["Subject"] = gdf["Subject"].str.strip()
    gdf["Gender"] = gdf["Gender"].str.strip().str.lower()
    return {s: (1 if g == "male" else 0) for s, g in zip(gdf["Subject"], gdf["Gender"])}


class GenderActionVideoDataset(ActionVideoDataset):
    """ActionVideoDataset that also returns each clip's participant gender (0=F, 1=M).

    Clips whose participant has no known gender are dropped at construction time so the
    sensitive attribute u is always defined.
    """

    def __init__(self, df: pd.DataFrame, label2idx: dict, gender_map: dict,
                 augment: bool = False, camera: str = DATA_CAMERA, video_type: str = DATA_VIDEO_TYPE):
        super().__init__(df, label2idx, augment, camera, video_type)
        kept = []
        for rec in self.records:
            subject = Path(rec["video_path"]).stem.split("_")[0]
            if subject in gender_map:
                rec["subject"] = subject
                rec["gender"] = gender_map[subject]
                kept.append(rec)
            else:
                warnings.warn(f"No gender label for {subject}; dropping its clips.")
        self.records = kept

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        item["gender"] = torch.tensor(self.records[idx]["gender"], dtype=torch.long)
        return item


def build_tappfl_datasets(csv_path=CSV_PATH, seed=SEED, video_type=DATA_VIDEO_TYPE,
                          camera=DATA_CAMERA, gender_csv=GENDER_CSV):
    """Gender-aware variant of build_federated_datasets for TAPPFL.

    Uses the same 70/15/15 participant split (subject-per-client kept). Each training
    subject is one client dataset; val/test stay centralized held-out sets. Every dataset
    yields the per-clip sensitive attribute `gender` (0=F, 1=M) in addition to
    `pixel_values` and the action `labels`.

    Returns: (client_datasets, train_eval_ds, val_ds, test_ds, label2idx)
      client_datasets: dict[subject_id -> GenderActionVideoDataset]  (augment=True)
      train_eval_ds:   pooled training subjects, augment=False (for frozen-encoder probes)
    """
    df = load_filtered_df(csv_path, video_type, camera)
    train_subjects, val_subjects, test_subjects = _split_subjects(df, seed)

    label2idx = {label: i for i, label in enumerate(LABEL_NAMES)}
    gender_map = load_gender_map(gender_csv)

    client_datasets = {}
    for subject in sorted(train_subjects):
        if subject not in gender_map:
            warnings.warn(f"No gender label for client {subject}; skipping this client.")
            continue
        sub_df = df[df["Subject"] == subject]
        ds = GenderActionVideoDataset(sub_df, label2idx, gender_map, augment=True,
                                      camera=camera, video_type=video_type)
        if len(ds) > 0:
            client_datasets[subject] = ds

    # Deterministic (augment=False) view of the SAME training subjects, used only for the
    # frozen-encoder privacy/utility probes (seen-client gender leakage + retrained head).
    train_df = df[df["Subject"].isin(train_subjects)]
    train_eval_ds = GenderActionVideoDataset(train_df, label2idx, gender_map, augment=False,
                                             camera=camera, video_type=video_type)
    val_df = df[df["Subject"].isin(val_subjects)]
    test_df = df[df["Subject"].isin(test_subjects)]
    val_ds = GenderActionVideoDataset(val_df, label2idx, gender_map, augment=False,
                                      camera=camera, video_type=video_type)
    test_ds = GenderActionVideoDataset(test_df, label2idx, gender_map, augment=False,
                                       camera=camera, video_type=video_type)

    n_client_clips = sum(len(ds) for ds in client_datasets.values())
    print(f"TAPPFL — clients: {len(client_datasets)}, val subjects: {len(val_subjects)}, "
          f"test subjects: {len(test_subjects)}")
    print(f"Clips   — clients(total): {n_client_clips}, val: {len(val_ds)}, test: {len(test_ds)}")

    return client_datasets, train_eval_ds, val_ds, test_ds, label2idx
