from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import pipeline

from src.config import VIDEO_ROOTS, DATA_DIR

OUTPUT_CSV  = DATA_DIR / "participant_genders.csv"
MODEL_ID    = "rizvandwiki/gender-classification"
NUM_SAMPLES = 10


def get_video_path(subject: str) -> Path:
    n = int(subject[1:])
    if n <= 26:
        folder = 1
    elif n <= 51:
        folder = 2
    elif n <= 75:
        folder = 3
    else:
        folder = 4
    return VIDEO_ROOTS[folder] / f"{subject}_FrontWeb_Action.mp4"


def sample_frames(video_path: Path, n: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def majority_vote(predictions: list[dict]) -> tuple[str, float]:
    votes = Counter(p["label"].lower() for p in predictions)
    winner = votes.most_common(1)[0][0]
    scores = [p["score"] for p in predictions if p["label"].lower() == winner]
    return winner, round(sum(scores) / len(scores), 4)


def main():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading gender classifier on device: {'cuda:0' if device == 0 else 'cpu'}")
    classifier = pipeline("image-classification", model=MODEL_ID, device=device)

    records = []
    for n in range(1, 78):
        subject = f"P{n:03d}"
        video_path = get_video_path(subject)
        print(f"[{n:02d}/77] {subject} ...", end=" ", flush=True)

        if not video_path.exists():
            print("MISSING — skipped")
            records.append({"Subject": subject, "Gender": "UNKNOWN", "Confidence": 0.0})
            continue

        try:
            frames = sample_frames(video_path, NUM_SAMPLES)
            if not frames:
                raise RuntimeError("No frames could be read")
            # Run classifier on each frame; each call returns list sorted by score desc
            per_frame_preds = [classifier(frame)[0] for frame in frames]
            gender, confidence = majority_vote(per_frame_preds)
            vote_summary = Counter(p["label"].lower() for p in per_frame_preds)
            print(f"{gender} ({confidence:.4f})  votes={dict(vote_summary)}")
            records.append({"Subject": subject, "Gender": gender, "Confidence": confidence})
        except Exception as exc:
            print(f"ERROR: {exc}")
            records.append({"Subject": subject, "Gender": "UNKNOWN", "Confidence": 0.0})

    df = pd.DataFrame(records, columns=["Subject", "Gender", "Confidence"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUTPUT_CSV}")
    print("\nGender distribution:")
    print(df["Gender"].value_counts().to_string())
    unknown = (df["Gender"] == "UNKNOWN").sum()
    if unknown:
        print(f"\nWARNING: {unknown} participant(s) could not be classified.")


if __name__ == "__main__":
    main()
