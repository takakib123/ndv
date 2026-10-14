"""Redaction operators applied to decoded clips before attribute inference.

The redaction-conditioned leakage metric feeds *occluded* clips through a frozen encoder and
retrains the gender probe on the resulting features. Leakage that survives redaction is not
being read off the pixels, so it is attributable to what the encoder memorized.

A `Redactor` is a (region, mode, severity) triple. `apply()` takes the (T, H, W, 3) uint8
array that `read_frames_cached` returns -- i.e. frames already downscaled to
CACHE_SHORT_EDGE -- and paints over it, so nothing needs re-decoding and the frame cache is
never invalidated. Region boxes come from `src.data.pose` in ORIGINAL video coordinates and
are rescaled here to whatever resolution the cached frames are at.

Regions:
  face / head_shoulders / torso / arms / person   pose-derived, painted inside the boxes
  background                                      pose-derived, painted OUTSIDE the person box
  grid_<i>_<j>                                    cell (i, j) of a k x k partition of the part
                                                  of the frame the encoder actually sees
  sham_<region>                                   the same number of ENCODER-VISIBLE pixels as
                                                  <region>, taken off the occupant
  none                                            identity

`sham` is the control that makes the metric meaningful. The encoder was trained on clean
video, so ANY occlusion is distribution shift; only the leakage drop BEYOND the sham
condition is attributable to the content of the region that was covered.

Geometry is defined relative to the ENCODER VIEW, not the raw frame. The VideoMAE processor
resizes the short edge to 224 and center-crops 224x224, so on 16:9 video the encoder never
sees the left/right ~22% of the frame (x < 420 or x > 1500 at 1920x1080). A box placed there
is a no-op for the model. An earlier version slid the sham box sideways off the occupant,
which put it mostly in that cropped margin (only ~27% of the face sham was visible, and it
was fully invisible in ~35% of frames), so "face beats sham" compared a real occlusion with
a mostly invisible one. A same-size box fits beside the occupant inside the view in only
~19% of frames, so the sham is now an area-matched pixel mask instead of a moved box.

Modes are all sized relative to the region's own width, so a condition means the same thing
regardless of what resolution the frames happen to be cached at.
"""

import re
import zlib
from functools import lru_cache

import cv2
import numpy as np

from src import config
from src.config import DATA_CAMERA, DATA_VIDEO_TYPE, REDACTION_GRID
from src.data.pose import get_subject_pose, REGIONS

MODES = ("blur", "black", "pixelate")

# Bump whenever what a condition paints changes (region geometry, sham or grid semantics), so
# features cached under the old definition are never silently reused.
#   v2: encoder-view-relative grid; area-matched, encoder-visible sham mask.
#   v3: BODY_25B column remap in pose.py; head anchor = centre of the face-joint extent.
REDACTION_VERSION = 3

# Mode strength at severity s, as a fraction of the region width W_r:
#   blur     Gaussian sigma  = s * STRENGTH_FRAC * W_r
#   pixelate mosaic block    = s * STRENGTH_FRAC * W_r
# At s=1 that is a half-region-wide kernel, which destroys the region outright; the low end of
# the severity grid (0.125) is a visible but mild degradation.
STRENGTH_FRAC = 0.5

_GRID_RE = re.compile(r"^grid_(\d+)_(\d+)$")


@lru_cache(maxsize=512)
def original_frame_size(video_path: str):
    """(width, height) of the source video -- boxes are in this coordinate frame."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w or 1920, h or 1080)


@lru_cache(maxsize=16)
def encoder_view_box(orig_size):
    """(x0, y0, x1, y1) of the region the encoder actually sees, in ORIGINAL video coordinates.

    Mirrors the image processor: resize the short edge to `shortest_edge`, then center-crop to
    `crop_size`. Anything outside this box is discarded before the model.
    """
    from src.data.dataset import get_image_processor
    p = get_image_processor()
    ow, oh = orig_size
    scale = p.size["shortest_edge"] / min(ow, oh)
    cw, ch = p.crop_size["width"] / scale, p.crop_size["height"] / scale
    return ((ow - cw) / 2, (oh - ch) / 2, (ow + cw) / 2, (oh + ch) / 2)


def cache_tag() -> str:
    """Hash of everything that determines redacted pixels, for feature-cache keys.

    Covers the code version plus every config knob the boxes are built from, so changing a
    box size in config.py invalidates cached redacted features instead of reusing them.
    """
    knobs = (REDACTION_VERSION, config.POSE_KEYPOINTS, config.POSE_CONF_MIN,
             config.FACE_HALF_W, config.FACE_HALF_H, config.FACE_CENTER_DY,
             config.TORSO_HALF_W, config.ARM_HALF, config.FALLBACK_DILATE,
             config.REDACTION_GRID, STRENGTH_FRAC, config.CACHE_SHORT_EDGE)
    return f"{zlib.crc32(repr(knobs).encode()):08x}"


def _gaussian_fast(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur approximated by three box passes.

    A true `cv2.GaussianBlur` at the sigmas this metric needs (half a region width -- hundreds
    of pixels at severity 1) costs O(n * ksize) and makes the sweep hours long. Box filtering
    is O(n) regardless of kernel size, and three passes converge to a Gaussian by the central
    limit theorem, which is indistinguishable here and orders of magnitude faster.
    """
    k = int(2 * round(sigma) + 1)                    # box width ~ 2*sigma, forced odd
    k = max(3, min(k, 2 * max(frame.shape[:2]) + 1))
    out = frame
    for _ in range(3):
        out = cv2.blur(out, (k, k))
    return out


def _transform_frame(frame: np.ndarray, mode: str, width: float) -> np.ndarray:
    """Whole-frame version of the effect; the mask decides where it actually lands."""
    if mode == "black":
        return np.zeros_like(frame)

    k = max(1.0, STRENGTH_FRAC * width)
    if mode == "blur":
        return _gaussian_fast(frame, k)

    if mode == "pixelate":
        h, w = frame.shape[:2]
        small = (max(1, int(round(w / k))), max(1, int(round(h / k))))
        down = cv2.resize(frame, small, interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)

    raise ValueError(f"Unknown redaction mode: {mode!r} (expected one of {MODES})")


class Redactor:
    """(region, mode, severity) occlusion applied to a clip's cached frames."""

    def __init__(self, region: str, mode: str = "black", severity: float = 1.0,
                 seed: int = 0, camera: str = DATA_CAMERA, video_type: str = DATA_VIDEO_TYPE):
        self.region = region
        self.mode = mode
        self.severity = float(severity)
        self.seed = seed
        self.camera = camera
        self.video_type = video_type
        self.last_tier = None      # evidence tier of the most recent lookup (see pose.py)
        self.last_sham_shortfall = 0.0   # fraction of a sham's target area that found no free room

        if region != "none":
            self._validate(region)
        if mode not in MODES:
            raise ValueError(f"Unknown redaction mode: {mode!r} (expected one of {MODES})")

    @staticmethod
    def _validate(region: str):
        base = region[len("sham_"):] if region.startswith("sham_") else region
        if base in REGIONS or _GRID_RE.match(base):
            return
        raise ValueError(f"Unknown redaction region: {region!r}")

    @property
    def is_identity(self) -> bool:
        # `black` at severity 0 would still paint, so severity gates every mode uniformly.
        return self.region == "none" or self.severity <= 0.0

    def key(self) -> str:
        """Short deterministic id, used for feature-cache filenames and plot labels."""
        if self.is_identity:
            return "none"
        return f"{self.region}_{self.mode}_s{self.severity:g}"

    # -- box resolution ----------------------------------------------------

    def _boxes_for_frame(self, subject, pose_frame, clip, orig_size):
        """(boxes, invert, sham) in ORIGINAL video coordinates for one frame.

        For a sham region the boxes are those of the TARGET region; `apply` turns them into an
        area-matched mask elsewhere in the encoder view.
        """
        ow, oh = orig_size
        region = self.region

        sham = region.startswith("sham_")
        if sham:
            region = region[len("sham_"):]

        grid = _GRID_RE.match(region)
        if grid:
            # Partition the encoder view, not the raw frame: on 16:9 video the outer columns
            # of a full-frame grid are ~88% cropped away before the model sees them.
            i, j = int(grid.group(1)), int(grid.group(2))
            vx0, vy0, vx1, vy1 = encoder_view_box(orig_size)
            cw, ch = (vx1 - vx0) / REDACTION_GRID, (vy1 - vy0) / REDACTION_GRID
            boxes, invert = [(vx0 + j * cw, vy0 + i * ch,
                              vx0 + (j + 1) * cw, vy0 + (i + 1) * ch)], False
            self.last_tier = 0     # grid cells need no pose at all
        else:
            sp = get_subject_pose(subject, self.camera, self.video_type, (ow, oh))
            boxes, tier, invert = sp.boxes(pose_frame, region, clip)
            self.last_tier = tier

        return boxes, invert, sham

    def _sham_mask(self, subject, pose_frame, clip, target, orig_size, scale, shape):
        """Area-matched control mask: as many ENCODER-VISIBLE pixels as `target` covers, taken
        from the visible frame outside the occupant, farthest-from-the-person first.

        Matching the visible area (not the box shape) is what makes the control fair -- the
        encoder sees the same amount of occlusion in both conditions, so only WHAT was covered
        differs. Taking the pixels farthest from the occupant keeps the control off the body
        even though the generous person box leaves little free room inside the view. Fully
        deterministic, so it is identical across conditions, checkpoints and workers.
        """
        h, w = shape
        sx, sy = scale

        def to_mask(box):
            x0, y0, x1, y1 = box
            m = np.zeros((h, w), dtype=bool)
            m[int(round(y0 * sy)):int(round(y1 * sy)), int(round(x0 * sx)):int(round(x1 * sx))] = True
            return m

        view = to_mask(encoder_view_box(orig_size))
        sp = get_subject_pose(subject, self.camera, self.video_type, orig_size)
        occupant = to_mask(sp.person_box(pose_frame, clip)[0])

        n = int((target & view).sum())
        free = view & ~occupant
        if n == 0 or not free.any():
            self.last_sham_shortfall = float(n > 0)
            return np.zeros((h, w), dtype=bool)

        # Distance of every pixel to the nearest occupant pixel (0 on the occupant).
        dist = cv2.distanceTransform((~occupant).astype(np.uint8), cv2.DIST_L2, 3)
        cand = np.flatnonzero(free)
        take = cand[np.argsort(-dist.ravel()[cand], kind="stable")[:n]]
        mask = np.zeros(h * w, dtype=bool)
        mask[take] = True
        self.last_sham_shortfall = 1.0 - len(take) / n
        return mask.reshape(h, w)

    # -- application -------------------------------------------------------

    def apply(self, frames: np.ndarray, subject: str, frame_indices, video_path: str) -> np.ndarray:
        """Redact a clip.

        frames         (T, H, W, 3) uint8, already downscaled to the frame-cache resolution
        frame_indices  the 0-based cv2 positions those frames were decoded from
        """
        if self.is_identity:
            return frames

        orig_size = original_frame_size(video_path)
        ow, oh = orig_size
        _, h, w, _ = frames.shape
        sx, sy = w / ow, h / oh

        # Pose CSVs number frames from 1; sample_frames yields 0-based cv2 positions. The clip
        # range lets the box anchor be a robust median over this clip rather than a per-frame
        # keypoint that may be spurious.
        clip = (int(min(frame_indices)) + 1, int(max(frame_indices)) + 1)

        out = frames.copy()
        for t, idx in enumerate(frame_indices):
            boxes, invert, sham = self._boxes_for_frame(subject, int(idx) + 1, clip, orig_size)

            scaled = []
            for (x0, y0, x1, y1) in boxes:
                sb = (int(round(x0 * sx)), int(round(y0 * sy)),
                      int(round(x1 * sx)), int(round(y1 * sy)))
                if sb[2] > sb[0] and sb[3] > sb[1]:
                    scaled.append(sb)
            if not scaled:
                continue

            mask = np.zeros((h, w), dtype=bool)
            for (x0, y0, x1, y1) in scaled:
                mask[y0:y1, x0:x1] = True
            if invert:
                mask = ~mask
            if sham and not invert:
                mask = self._sham_mask(subject, int(idx) + 1, clip, mask, orig_size,
                                       (sx, sy), (h, w))
            # (The sham of an inverted region -- background -- stays the person box itself:
            # the complement has no "somewhere else". It is not in the default sweep.)
            if not mask.any():
                continue

            # Strength scales with the region's own width, so a severity means the same thing
            # for a face box and for a whole-person box. A sham uses its target's width.
            ref_width = float(np.mean([b[2] - b[0] for b in scaled])) * self.severity
            transformed = _transform_frame(out[t], self.mode, ref_width)
            out[t] = np.where(mask[..., None], transformed, out[t])

        return out


def parse_condition(spec: str, seed: int = 0) -> Redactor:
    """Build a Redactor from a CLI string: 'none', 'face_black', 'face_blur_s0.25', ...

    Region names themselves contain underscores, so the mode is located by scanning for a
    known mode token rather than by splitting on the last separator.
    """
    if spec == "none":
        return Redactor("none")

    parts = spec.split("_")
    severity = 1.0
    if parts[-1].startswith("s") and _is_float(parts[-1][1:]):
        severity = float(parts[-1][1:])
        parts = parts[:-1]

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in MODES:
            region = "_".join(parts[:i])
            return Redactor(region, parts[i], severity, seed=seed)
    raise ValueError(f"Could not parse condition {spec!r}; expected <region>_<mode>[_s<severity>]")


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
