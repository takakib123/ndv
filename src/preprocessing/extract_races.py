"""Derive a per-participant RACE label by polling two independent model families.

Mirrors preprocessing/extract_genders.py (sample frames -> classify -> vote) but with a
stronger evidence base, because race is a harder call off a dashcam frame than gender:

  evidence   7 images per participant -- the prepared StillFacesCropped portrait plus 6
             pose-cropped faces sampled across the participant's FrontWeb_Action video, so
             the poll sees real variation in lighting/pose/background, not one image resampled.
  voter A    a multimodal VLM (Qwen2.5-VL) asked under 3 prompt phrasings. Instead of parsing
             free text, each of the 4 candidate answers is scored by forced-decode logprob and
             softmaxed -- a probability vector per pass, and parse failures cannot happen.
  voter B    a specialized face-attribute ViT (cledoux42/Ethnicity_Test_v003), whose label set
             maps almost 1:1 onto the taxonomy.
  fusion     average each family's per-pass probability vectors, then fuse the families
             equally. Disagreement between families is the main uncertainty flag.

CAVEAT -- these are PERCEIVED-RACE annotations inferred by models from facial appearance, not
self-reported demographics. They exist to *measure* attribute leakage (the point of this repo)
and must not be reported as ground-truth participant demographics. Every label with
Uncertain=1 should be reviewed by hand before use; as with gender, the reviewed file
(participant_races_manual.csv) is what downstream code should read.

Usage:
    python -m src.preprocessing.extract_races                  # full 77-participant run
    python -m src.preprocessing.extract_races --limit 3        # smoke test on P001-P003
    python -m src.preprocessing.extract_races --dump-crops     # also write the sampled images
    python -m src.preprocessing.extract_races --gender-check   # validate crops against gender
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import VIDEO_ROOTS, DATA_DIR, ARTIFACTS_DIR, GENDER_CSV
from src.data.pose import get_subject_pose

RACE_CLASSES = ("white", "black", "latino", "asian")

OUTPUT_CSV   = DATA_DIR / "participant_races.csv"   # sibling _votes.json / _review.csv follow --out
CROP_DIR     = ARTIFACTS_DIR / "outputs" / "race_crops"

STILLS_DIR   = DATA_DIR / "StillFacesCropped" / "StillFacesCropped"

VLM_MODEL    = "Qwen/Qwen2.5-VL-7B-Instruct"
CLS_MODEL    = "cledoux42/Ethnicity_Test_v003"
GENDER_MODEL = "rizvandwiki/gender-classification"   # the model that made the gender labels

N_VIDEO_FRAMES = 6
N_SUBJECTS     = 77

# Voter B's label space -> ours. 'indian' has no home in the 4-class taxonomy: its mass is
# dropped and the rest renormalized, but the dropped mass is reported as a diagnostic, since a
# participant the classifier reads as Indian is exactly the kind of forced call needing review.
CLS_LABEL_MAP = {
    "african":   "black",
    "asian":     "asian",
    "caucasian": "white",
    "hispanic":  "latino",
    "indian":    None,
}

# Three phrasings. The taxonomy is stated explicitly in each so the model answers inside it
# rather than volunteering a label we cannot score.
PROMPTS = (
    "Look at this person's face. Which of these four categories best describes their race or "
    "ethnicity: white, black, latino, or asian? Answer with exactly one word.",
    "You are annotating a face dataset. Classify the perceived race of the person shown. "
    "The only allowed labels are: white, black, latino, asian. Reply with the label alone.",
    "Based on facial appearance alone, is this person best described as white, black, latino, "
    "or asian? One word only.",
)

# Uncertainty gates -- any one of these trips the review flag.
MIN_CONFIDENCE   = 0.55   # fused winning probability
MIN_VLM_AGREE    = 0.70   # fraction of VLM passes agreeing with the VLM's own argmax
MAX_INDIAN_MASS  = 0.40   # mean out-of-taxonomy mass from voter B


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

def get_video_path(subject: str) -> Path:
    """Same folder layout mapping as extract_genders.get_video_path."""
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


def still_face_path(subject: str) -> Path:
    return STILLS_DIR / f"{subject}_Face.jpg"


def _square_from_face_box(box, frame_w: int, frame_h: int):
    """config.FACE_* sizes a tall box (~307x518) built to *redact* the head, hair included.

    For classification we want a square crop comparable to the 273x264 stills, so keep the box
    centre and take a square of side = the box width.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = (x1 - x0) / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    # Shift (don't shrink) a square that runs off the frame, so the aspect ratio holds.
    if x0 < 0: x1, x0 = x1 - x0, 0
    if y0 < 0: y1, y0 = y1 - y0, 0
    if x1 > frame_w: x0, x1 = x0 - (x1 - frame_w), frame_w
    if y1 > frame_h: y0, y1 = y0 - (y1 - frame_h), frame_h
    return max(0, x0), max(0, y0), min(frame_w, x1), min(frame_h, y1)


def sample_video_faces(subject: str, n: int) -> list[tuple[str, Image.Image]]:
    """n face crops spread across the participant's video, as (source_tag, image).

    Only frames whose Head keypoint actually resolved are sampled, so every crop is anchored
    at pose.TIER_PER_FRAME instead of falling back to a subject-median box. Pose rows are
    1-based video frame numbers while cv2 seeks 0-based, the same offset redaction.py:214 applies.
    """
    video_path = get_video_path(subject)
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    sp = get_subject_pose(subject, frame_size=(frame_w, frame_h))
    clip = (1, max(1, total))

    if sp.pose is not None and len(sp.pose):
        ok = np.isfinite(sp.pose["HeadX"].to_numpy())
        usable = sp.pose.index.to_numpy()[ok]
        usable = usable[(usable >= 1) & (usable <= total)]
    else:
        usable = np.array([], dtype=int)

    if len(usable) >= n:
        pose_frames = usable[np.linspace(0, len(usable) - 1, n).astype(int)]
    else:
        # No usable pose: fall back to evenly spaced frames; face_box resolves them through
        # its own subject-median tier.
        pose_frames = (np.linspace(0, total - 1, n).astype(int) + 1) if total > 0 else []

    out = []
    for pf in pose_frames:
        pf = int(pf)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pf - 1)
        ret, frame = cap.read()
        if not ret:
            continue
        box, _tier = sp.face_box(pf, clip)
        x0, y0, x1, y1 = _square_from_face_box(box, frame_w, frame_h)
        if x1 - x0 < 32 or y1 - y0 < 32:
            continue
        crop = frame[y0:y1, x0:x1]
        out.append((f"frame{pf}", Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out


def collect_images(subject: str, n_video: int) -> list[tuple[str, Image.Image]]:
    """The prepared portrait first, then the sampled in-cabin crops."""
    images = []
    still = still_face_path(subject)
    if still.exists():
        images.append(("still", Image.open(still).convert("RGB")))
    images.extend(sample_video_faces(subject, n_video))
    return images


# ---------------------------------------------------------------------------
# Voter A -- multimodal VLM, forced-decode scoring over the 4 candidates
# ---------------------------------------------------------------------------

class VLMVoter:
    """Scores each candidate answer by summed token logprob, then softmaxes over the four.

    Generation + regex parsing would leave the model free to answer off-taxonomy ("Middle
    Eastern", "I can't tell"); scoring fixed continuations cannot fail to produce a vector.
    """

    def __init__(self, model_id: str = VLM_MODEL, temperature: float = 1.0, device: str = "cuda"):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.temperature = float(temperature)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
        ).eval()

        tok = self.processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"   # scoring reads candidate tokens off the real length
        self.cand_lens = [len(tok(c, add_special_tokens=False).input_ids) for c in RACE_CLASSES]

    def _chat_prefix(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str) -> np.ndarray:
        """Probability vector over RACE_CLASSES for one (image, prompt) pass."""
        prefix = self._chat_prefix(prompt)
        texts = [prefix + c for c in RACE_CLASSES]
        inputs = self.processor(text=texts, images=[image] * len(RACE_CLASSES),
                                return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        logits = self.model(**inputs).logits.float()
        ids, mask = inputs["input_ids"], inputs["attention_mask"]

        scores = []
        for i, n_c in enumerate(self.cand_lens):
            end = int(mask[i].sum())               # real length, padding excluded
            tgt = ids[i, end - n_c:end]            # the candidate's own tokens
            lp = torch.log_softmax(logits[i, end - n_c - 1:end - 1], dim=-1)
            scores.append(lp.gather(1, tgt[:, None]).sum().item())

        s = np.asarray(scores, dtype=np.float64) / max(self.temperature, 1e-6)
        s -= s.max()
        p = np.exp(s)
        return p / p.sum()


# ---------------------------------------------------------------------------
# Voter B -- specialized face-attribute classifier
# ---------------------------------------------------------------------------

class ClassifierVoter:
    """Wraps the same transformers image-classification pipeline extract_genders.py uses."""

    def __init__(self, model_id: str = CLS_MODEL, device: int = 0):
        from transformers import pipeline
        self.pipe = pipeline("image-classification", model=model_id, device=device)

    def score(self, image: Image.Image) -> tuple[np.ndarray, float]:
        """(probability vector over RACE_CLASSES, out-of-taxonomy mass)."""
        preds = self.pipe(image, top_k=None)
        vec = np.zeros(len(RACE_CLASSES), dtype=np.float64)
        dropped = 0.0
        for p in preds:
            mapped = CLS_LABEL_MAP.get(p["label"].strip().lower())
            if mapped is None:
                dropped += float(p["score"])
            else:
                vec[RACE_CLASSES.index(mapped)] += float(p["score"])
        total = vec.sum()
        return (vec / total if total > 0 else np.full(len(RACE_CLASSES), 1.0 / len(RACE_CLASSES))), dropped


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse(vlm_passes: list[np.ndarray], cls_passes: list[np.ndarray], w_vlm: float):
    """Average within each family, then across families at weight w_vlm.

    Returns (fused, vlm_mean|None, cls_mean|None). A family that cast no votes is skipped and
    the other carries full weight, so --skip-vlm / a missing model still yields a label.
    """
    vlm_mean = np.mean(vlm_passes, axis=0) if vlm_passes else None
    cls_mean = np.mean(cls_passes, axis=0) if cls_passes else None

    if vlm_mean is not None and cls_mean is not None:
        fused = w_vlm * vlm_mean + (1.0 - w_vlm) * cls_mean
    elif vlm_mean is not None:
        fused = vlm_mean
    elif cls_mean is not None:
        fused = cls_mean
    else:
        return None, None, None
    return fused / fused.sum(), vlm_mean, cls_mean


def vlm_agreement(vlm_passes: list[np.ndarray]) -> float:
    """Fraction of individual passes whose argmax matches the family's mean argmax."""
    if not vlm_passes:
        return float("nan")
    winner = int(np.argmax(np.mean(vlm_passes, axis=0)))
    return float(np.mean([int(np.argmax(p)) == winner for p in vlm_passes]))


# ---------------------------------------------------------------------------
# Crop validation via the known-good gender classifier
# ---------------------------------------------------------------------------

def run_gender_check(records: list[dict]) -> None:
    """Sanity-check the evidence pipeline, not the race models.

    The same 7 images are pushed through the classifier that produced the gender labels. If
    those images reproduce the manual gender labels, they contain a usable face; a low score
    means the sampling or the crop geometry is broken, and the race labels are untrustworthy
    for reasons that have nothing to do with the race models.
    """
    if not GENDER_CSV.exists():
        print(f"\nGender cross-check skipped: {GENDER_CSV} not found")
        return

    gdf = pd.read_csv(GENDER_CSV)
    gdf.columns = gdf.columns.str.strip()
    gdf = gdf.dropna(subset=["Subject", "Gender"])
    truth = {s.strip(): g.strip().lower() for s, g in zip(gdf["Subject"], gdf["Gender"])}

    print(f"\nCrop validation: {GENDER_MODEL} over the same images ...")
    voter = ClassifierVoter.__new__(ClassifierVoter)
    from transformers import pipeline
    voter.pipe = pipeline("image-classification", model=GENDER_MODEL,
                          device=0 if torch.cuda.is_available() else -1)

    hits = total = 0
    for rec in records:
        subject = rec["Subject"]
        if subject not in truth or not rec.get("_images"):
            continue
        votes = []
        for _tag, img in rec["_images"]:
            votes.append(voter.pipe(img, top_k=1)[0]["label"].strip().lower())
        pred = max(set(votes), key=votes.count)
        hits += int(pred == truth[subject])
        total += 1

    if total:
        acc = hits / total
        verdict = "OK" if acc >= 0.90 else "LOW -- inspect the crops before trusting the race labels"
        print(f"Gender agreement on the sampled images: {hits}/{total} = {acc:.3f}  [{verdict}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Poll a VLM and a face-attribute classifier for participant race")
    p.add_argument("--limit", type=int, default=None, help="only process the first N participants (smoke test)")
    p.add_argument("--n-video-frames", type=int, default=N_VIDEO_FRAMES,
                   help="video face crops per participant; VLM passes = (1 + this) x %d prompts" % len(PROMPTS))
    p.add_argument("--w-vlm", type=float, default=0.5, help="weight on the VLM family when fusing (0..1)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="softmax temperature over the VLM's candidate logprobs (>1 flattens, <1 sharpens)")
    p.add_argument("--vlm-model", default=VLM_MODEL)
    p.add_argument("--cls-model", default=CLS_MODEL)
    p.add_argument("--skip-vlm", action="store_true", help="classifier only")
    p.add_argument("--skip-cls", action="store_true", help="VLM only")
    p.add_argument("--dump-crops", action="store_true", help="write the sampled images to artifacts/outputs/race_crops/")
    p.add_argument("--gender-check", action="store_true", help="validate the crops against the manual gender labels")
    p.add_argument("--out", type=Path, default=OUTPUT_CSV)
    return p.parse_args()


def main():
    args = parse_args()
    if args.skip_vlm and args.skip_cls:
        raise SystemExit("--skip-vlm and --skip-cls together leave no voters")

    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'cuda:0' if device == 0 else 'cpu'}")

    vlm = None if args.skip_vlm else VLMVoter(args.vlm_model, args.temperature,
                                              "cuda" if device == 0 else "cpu")
    cls = None if args.skip_cls else ClassifierVoter(args.cls_model, device)
    if vlm:
        print(f"VLM: {args.vlm_model}  ({(1 + args.n_video_frames) * len(PROMPTS)} passes/participant)")
    if cls:
        print(f"Classifier: {args.cls_model}")

    n_total = min(args.limit or N_SUBJECTS, N_SUBJECTS)
    records, audit = [], {}

    for n in range(1, n_total + 1):
        subject = f"P{n:03d}"
        print(f"[{n:02d}/{n_total}] {subject} ...", end=" ", flush=True)

        try:
            images = collect_images(subject, args.n_video_frames)
            if not images:
                raise RuntimeError("no usable images (missing still and video)")

            if args.dump_crops:
                out_dir = CROP_DIR / subject
                out_dir.mkdir(parents=True, exist_ok=True)
                for tag, img in images:
                    img.save(out_dir / f"{tag}.jpg", quality=95)

            vlm_passes, cls_passes, indian_mass, passes = [], [], [], []
            for tag, img in images:
                if cls is not None:
                    vec, dropped = cls.score(img)
                    cls_passes.append(vec)
                    indian_mass.append(dropped)
                    passes.append({"model": "cls", "image": tag, "prompt": None,
                                   "probs": dict(zip(RACE_CLASSES, np.round(vec, 4).tolist())),
                                   "out_of_taxonomy": round(dropped, 4)})
                if vlm is not None:
                    for pi, prompt in enumerate(PROMPTS):
                        vec = vlm.score(img, prompt)
                        vlm_passes.append(vec)
                        passes.append({"model": "vlm", "image": tag, "prompt": pi,
                                       "probs": dict(zip(RACE_CLASSES, np.round(vec, 4).tolist()))})

            fused, vlm_mean, cls_mean = fuse(vlm_passes, cls_passes, args.w_vlm)
            order = np.argsort(fused)[::-1]
            race = RACE_CLASSES[order[0]]
            confidence = float(fused[order[0]])
            margin = float(fused[order[0]] - fused[order[1]])

            agree = vlm_agreement(vlm_passes)
            mean_indian = float(np.mean(indian_mass)) if indian_mass else 0.0
            families_disagree = (
                vlm_mean is not None and cls_mean is not None
                and int(np.argmax(vlm_mean)) != int(np.argmax(cls_mean))
            )
            uncertain = int(
                confidence < MIN_CONFIDENCE
                or families_disagree
                or (vlm_passes and agree < MIN_VLM_AGREE)
                or mean_indian > MAX_INDIAN_MASS
            )

            records.append({
                "Subject": subject, "Race": race,
                "Confidence": round(confidence, 4),
                "VoteMargin": round(margin, 4),
                "Uncertain": uncertain,
                "_images": images,
                "_vlm": RACE_CLASSES[int(np.argmax(vlm_mean))] if vlm_mean is not None else "",
                "_cls": RACE_CLASSES[int(np.argmax(cls_mean))] if cls_mean is not None else "",
                "_agree": agree, "_indian": mean_indian, "_disagree": int(families_disagree),
            })
            audit[subject] = {
                "n_images": len(images),
                "fused": dict(zip(RACE_CLASSES, np.round(fused, 4).tolist())),
                "vlm_mean": dict(zip(RACE_CLASSES, np.round(vlm_mean, 4).tolist())) if vlm_mean is not None else None,
                "cls_mean": dict(zip(RACE_CLASSES, np.round(cls_mean, 4).tolist())) if cls_mean is not None else None,
                "vlm_agreement": None if np.isnan(agree) else round(agree, 4),
                "mean_out_of_taxonomy": round(mean_indian, 4),
                "passes": passes,
            }

            flag = "  FLAG" if uncertain else ""
            print(f"{race} ({confidence:.3f}, margin {margin:.3f}) "
                  f"vlm={records[-1]['_vlm'] or '-'} cls={records[-1]['_cls'] or '-'} "
                  f"imgs={len(images)}{flag}")

        except Exception as exc:
            print(f"ERROR: {exc}")
            records.append({"Subject": subject, "Race": "UNKNOWN", "Confidence": 0.0,
                            "VoteMargin": 0.0, "Uncertain": 1, "_images": [],
                            "_vlm": "", "_cls": "", "_agree": float("nan"),
                            "_indian": 0.0, "_disagree": 0})
            audit[subject] = {"error": str(exc)}

    if args.gender_check:
        run_gender_check(records)

    # ---- outputs ----------------------------------------------------------
    # The audit and review files ride along with --out, so a smoke test writing elsewhere
    # cannot clobber the real data/ outputs.
    votes_json = args.out.with_name(args.out.stem + "_votes.json")
    review_csv = args.out.with_name(args.out.stem + "_review.csv")

    cols = ["Subject", "Race", "Confidence", "VoteMargin", "Uncertain"]
    df = pd.DataFrame(records)[cols]
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows -> {args.out}")

    votes_json.write_text(json.dumps(audit, indent=2))
    print(f"Saved per-pass audit trail -> {votes_json}")

    review = pd.DataFrame([{
        "Subject": r["Subject"], "Race": r["Race"],
        "Confidence": r["Confidence"], "VoteMargin": r["VoteMargin"],
        "VLM": r["_vlm"], "Classifier": r["_cls"],
        "FamiliesDisagree": r["_disagree"],
        "VLMAgreement": "" if np.isnan(r["_agree"]) else round(r["_agree"], 3),
        "OutOfTaxonomyMass": round(r["_indian"], 3),
        "StillImage": str(still_face_path(r["Subject"])),
    } for r in records if r["Uncertain"]])
    review.to_csv(review_csv, index=False)
    print(f"Saved {len(review)} flagged participant(s) for review -> {review_csv}")

    print("\nRace distribution:")
    print(df["Race"].value_counts().to_string())
    n_flag = int(df["Uncertain"].sum())
    if n_flag:
        print(f"\n{n_flag}/{len(df)} participant(s) flagged Uncertain. Review {review_csv} and "
              f"save the corrected labels as annot/participant_races_manual.csv before use.")


if __name__ == "__main__":
    main()
