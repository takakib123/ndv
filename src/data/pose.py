"""Pose-derived region boxes for the redaction-based leakage metric.

Turns the per-frame OpenPose-25B keypoints in `data/Pose_OpenPose25b/` into pixel boxes for
semantic regions (face, torso, arms, whole person, background) in the ORIGINAL video
coordinate frame. `src.data.redaction` scales those boxes to whatever resolution the frames
were cached at and paints over them.

Three dataset facts drive the design:

1. The CSV headers are wrong. The files are OpenPose BODY_25B (COCO joint order: nose, eyes,
   ears, shoulders, elbows, wrists, hips, ..., upper-neck, head-top) but the columns carry
   BODY_25 names, so "RightShoulder" is really the right eye and "RightWrist" the right ear.
   `load_pose` renames columns by POSITION to the true joints. Of those, nose/eyes/ears and
   shoulders are reliable; elbows and wrists are essentially never detected (arms sit below
   the dash), so arm regions fall back to the flanks of the torso.
2. Pose-derived SCALE is unusable. Shoulder span collapses whenever the subject leans and one
   shoulder is occluded, swinging the implied body scale ~6x within a clip. Since the camera
   is fixed and everyone sits in the driver's seat, apparent head size is near-constant, so
   boxes are sized as fractions of the FRAME instead.
3. Per-frame pose can be missing (dropped rows, low-confidence nose). Boxes resolve through a
   three-tier fallback ladder and every lookup reports which tier it used, so the driver can
   log coverage and re-run restricted to tier-1 clips.

Boxes are deliberately generous. Under-covering the face would leave gender evidence visible
and silently inflate the "leakage that survives redaction" number, which is the one result
the metric exists to produce -- so the error is biased the safe way.
"""

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    POSE_ROOT, POSE_KEYPOINTS, POSE_CONF_MIN,
    FACE_HALF_W, FACE_HALF_H, FACE_CENTER_DY,
    TORSO_HALF_W, ARM_HALF, FALLBACK_DILATE,
    DATA_CAMERA, DATA_VIDEO_TYPE,
)

# Evidence tier a box was built from -- lower is better. Reported per clip so the driver can
# quantify how much of the metric rests on the fallback.
TIER_PER_FRAME = 1        # this frame's own head anchor (face joints)
TIER_CLIP_MEDIAN = 2      # clip's median Head (frame missing or its Head was an outlier)
TIER_SUBJECT_MEDIAN = 3   # subject's whole-video median, dilated (no pose anywhere in clip)

REGIONS = ("face", "head_shoulders", "torso", "arms", "person", "background")

FACE_JOINTS = ("Nose", "LEye", "REye", "LEar", "REar")

# True OpenPose BODY_25B joint order. The CSVs store 25 (X, Y, Conf) triples in this order
# under BODY_25 header names, so columns are renamed by position, never by header.
BODY_25B = (
    "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "UpperNeck", "HeadTop", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
)


def pose_path(subject: str, camera: str = DATA_CAMERA, video_type: str = DATA_VIDEO_TYPE) -> Path:
    return POSE_ROOT / f"{subject}_{camera}_{video_type}_OPo25B.csv"


@lru_cache(maxsize=256)
def load_pose(subject: str, camera: str = DATA_CAMERA, video_type: str = DATA_VIDEO_TYPE):
    """Per-frame keypoints for one clip file, indexed by the video's 1-based frame number.

    Returns None when the file is absent. Low-confidence coordinates become NaN so consumers
    can use nan-aware reductions instead of re-checking confidence everywhere.
    """
    path = pose_path(subject, camera, video_type)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    joints = [c[:-len("Conf")] for c in df.columns if c.endswith("Conf")]
    if len(joints) != len(BODY_25B):
        raise ValueError(f"{path.name}: expected {len(BODY_25B)} joints, found {len(joints)}")
    df = df.rename(columns={f"{old}{ax}": f"{new}{ax}"
                            for old, new in zip(joints, BODY_25B) for ax in ("X", "Y", "Conf")})
    out = pd.DataFrame(index=df["Frame"].astype(int))
    for kp in POSE_KEYPOINTS:
        ok = df[f"{kp}Conf"].to_numpy() > POSE_CONF_MIN
        out[f"{kp}X"] = np.where(ok, df[f"{kp}X"], np.nan)
        out[f"{kp}Y"] = np.where(ok, df[f"{kp}Y"], np.nan)

    # Head anchor. X is the centre of the visible face joints' horizontal extent rather than the
    # nose alone: in profile the nose sits at the FRONT edge of the head (P070 waving: nose
    # x=1187, ear x=930), so a nose-centred box covers the air in front of the face and leaves
    # half the face visible. Only the CENTRE moves -- box size stays frame-relative, because a
    # mask whose size tracks head size could itself leak gender to the probe (cf. ROAD, Rong et
    # al. 2022). Y stays the nose, which FACE_CENTER_DY was calibrated against; no nose -> no
    # per-frame anchor.
    face_x = out[[f"{k}X" for k in FACE_JOINTS]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN rows -> NaN, expected
        lo, hi = np.nanmin(face_x, axis=1), np.nanmax(face_x, axis=1)
    out["HeadX"] = (lo + hi) / 2
    out["HeadY"] = out["NoseY"]
    out.loc[out["NoseY"].isna(), "HeadX"] = np.nan
    return out[~out.index.duplicated(keep="first")]


class SubjectPose:
    """Region-box lookup for one (subject, camera, video_type), with the fallback ladder.

    Construct once per subject and reuse across clips; `load_pose` is memoized and the
    per-subject medians backing tier 3 are computed once here.
    """

    def __init__(self, subject: str, camera: str = DATA_CAMERA,
                 video_type: str = DATA_VIDEO_TYPE, frame_size=(1920, 1080)):
        self.subject = subject
        self.frame_w, self.frame_h = frame_size
        self.pose = load_pose(subject, camera, video_type)
        self._clip_cache = {}

        # Tier-3 anchor: whole-video median head, plus how far the head roams (5-95 pct), so
        # the fallback box can be widened to cover it.
        self._median_head = (0.59 * self.frame_w, 0.55 * self.frame_h)   # fixed-rig default
        self._head_spread = (0.0, 0.0)
        if self.pose is not None and len(self.pose):
            hx, hy = self.pose["HeadX"].to_numpy(), self.pose["HeadY"].to_numpy()
            if np.isfinite(hx).any():
                self._median_head = (float(np.nanmedian(hx)), float(np.nanmedian(hy)))
                self._head_spread = (
                    float(np.diff(np.nanpercentile(hx, [5, 95]))[0] / 2),
                    float(np.diff(np.nanpercentile(hy, [5, 95]))[0] / 2),
                )

    # -- anchors -----------------------------------------------------------

    def _clip_stats(self, clip):
        """(median_head_x, median_head_y, n_rows) over the pose rows inside one clip."""
        if clip in self._clip_cache:
            return self._clip_cache[clip]

        stats = None
        if self.pose is not None:
            lo, hi = clip
            sub = self.pose[(self.pose.index >= lo) & (self.pose.index <= hi)]
            if len(sub) and np.isfinite(sub["HeadX"].to_numpy()).any():
                stats = (float(np.nanmedian(sub["HeadX"])), float(np.nanmedian(sub["HeadY"])),
                         int(np.isfinite(sub["HeadX"].to_numpy()).sum()))
        self._clip_cache[clip] = stats
        return stats

    def _anchor(self, frame_idx: int, clip):
        """(x, y, extra_margin, tier) for the face centre of one frame."""
        stats = self._clip_stats(clip)
        if stats is None:
            # No pose anywhere in this clip: whole-video median, widened by how far the head
            # is known to travel.
            hx, hy = self._median_head
            return hx, hy, max(self._head_spread), TIER_SUBJECT_MEDIAN

        mx, my, _ = stats
        row = self._row(frame_idx)
        if row is not None and not np.isnan(row["HeadX"]):
            # Trust the per-frame keypoint unconditionally. An outlier gate was tried and had
            # to be removed: during TouchPassengerSeat the subject leans right out of the
            # driver's position, so the head legitimately sweeps ~700px across the clip and
            # the gate rejected the *upright* frames as outliers of their own clip median.
            # Overlays confirm the keypoint tracks the nose accurately even mid-lean.
            return float(row["HeadX"]), float(row["HeadY"]), 0.0, TIER_PER_FRAME
        return mx, my, 0.0, TIER_CLIP_MEDIAN

    def _row(self, frame_idx: int):
        if self.pose is None or frame_idx not in self.pose.index:
            return None
        return self.pose.loc[frame_idx]

    def _clip_box(self, box):
        x0, y0, x1, y1 = box
        return (max(0.0, x0), max(0.0, y0),
                min(float(self.frame_w), x1), min(float(self.frame_h), y1))

    # -- regions -----------------------------------------------------------

    def face_box(self, frame_idx: int, clip):
        """Generous box covering the head. Returns ((x0,y0,x1,y1), tier)."""
        hx, hy, margin, tier = self._anchor(frame_idx, clip)
        half_w = FACE_HALF_W * self.frame_w
        half_h = FACE_HALF_H * self.frame_h
        if tier == TIER_SUBJECT_MEDIAN:
            half_w = half_w * FALLBACK_DILATE + margin
            half_h = half_h * FALLBACK_DILATE + margin
        cy = hy + FACE_CENTER_DY * self.frame_h
        return self._clip_box((hx - half_w, cy - half_h, hx + half_w, cy + half_h)), tier

    def torso_box(self, frame_idx: int, clip):
        """Shoulders and chest down to the bottom of the frame (legs are never visible)."""
        hx, hy, margin, tier = self._anchor(frame_idx, clip)
        half_w = TORSO_HALF_W * self.frame_w + margin
        top = hy + 0.5 * FACE_HALF_H * self.frame_h       # just below the chin
        cx = hx
        row = self._row(frame_idx)
        if row is not None:
            xs = [row[f"{k}X"] for k in ("RShoulder", "LShoulder")]
            xs = [x for x in xs if not np.isnan(x)]
            if xs:
                cx = float(np.mean(xs))
        return self._clip_box((cx - half_w, top, cx + half_w, float(self.frame_h))), tier

    def head_shoulders_box(self, frame_idx: int, clip):
        (fx0, fy0, fx1, fy1), tier = self.face_box(frame_idx, clip)
        (tx0, _, tx1, _), _ = self.torso_box(frame_idx, clip)
        return self._clip_box((min(fx0, tx0), fy0, max(fx1, tx1),
                               fy1 + 0.25 * FACE_HALF_H * self.frame_h)), tier

    def arm_boxes(self, frame_idx: int, clip):
        """One box per visible elbow/wrist; falls back to the flanks of the torso."""
        hx, _, _, tier = self._anchor(frame_idx, clip)
        half = ARM_HALF * self.frame_w
        boxes = []
        row = self._row(frame_idx)
        if row is not None:
            for kp in ("RElbow", "LElbow", "RWrist", "LWrist"):
                x, y = row[f"{kp}X"], row[f"{kp}Y"]
                if not np.isnan(x):
                    boxes.append(self._clip_box((x - half, y - half, x + half, y + half)))
        if not boxes:
            (tx0, ty0, tx1, ty1), _ = self.torso_box(frame_idx, clip)
            w = (tx1 - tx0) * 0.35
            boxes = [self._clip_box((tx0 - w, ty0, tx0 + w, ty1)),
                     self._clip_box((tx1 - w, ty0, tx1 + w, ty1))]
            tier = max(tier, TIER_CLIP_MEDIAN)
        return boxes, tier

    def person_box(self, frame_idx: int, clip):
        """One box covering head, torso and arms -- the whole visible occupant."""
        (fx0, fy0, fx1, fy1), tier = self.face_box(frame_idx, clip)
        (tx0, ty0, tx1, ty1), _ = self.torso_box(frame_idx, clip)
        arms, _ = self.arm_boxes(frame_idx, clip)
        xs0 = [fx0, tx0] + [b[0] for b in arms]
        ys0 = [fy0, ty0] + [b[1] for b in arms]
        xs1 = [fx1, tx1] + [b[2] for b in arms]
        ys1 = [fy1, ty1] + [b[3] for b in arms]
        return self._clip_box((min(xs0), min(ys0), max(xs1), max(ys1))), tier

    def boxes(self, frame_idx: int, region: str, clip):
        """Region -> (boxes, tier, invert).

        `invert` marks regions defined as the complement of their boxes (background), which
        the redactor paints outside rather than inside.
        """
        if region == "face":
            b, t = self.face_box(frame_idx, clip)
            return [b], t, False
        if region == "head_shoulders":
            b, t = self.head_shoulders_box(frame_idx, clip)
            return [b], t, False
        if region == "torso":
            b, t = self.torso_box(frame_idx, clip)
            return [b], t, False
        if region == "arms":
            b, t = self.arm_boxes(frame_idx, clip)
            return b, t, False
        if region == "person":
            b, t = self.person_box(frame_idx, clip)
            return [b], t, False
        if region == "background":
            b, t = self.person_box(frame_idx, clip)
            return [b], t, True
        raise ValueError(f"Unknown pose region: {region!r} (expected one of {REGIONS})")


@lru_cache(maxsize=256)
def get_subject_pose(subject: str, camera: str = DATA_CAMERA,
                     video_type: str = DATA_VIDEO_TYPE, frame_size=(1920, 1080)) -> SubjectPose:
    return SubjectPose(subject, camera, video_type, frame_size)
