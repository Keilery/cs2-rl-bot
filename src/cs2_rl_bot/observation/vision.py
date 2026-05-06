"""Optional YOLOv8-based enemy detector + observation encoder.

`ultralytics` is an optional dependency (install with ``pip install -e '.[vision]'``).
If it is not installed, the detector is a no-op. The detector returns axis-aligned
bounding boxes in the same coordinate frame as the input image.

The :func:`encode_enemy_features` helper turns a list of :class:`Detection` into
a fixed-size float vector that the policy can consume directly. We expose the
top-K closest-to-crosshair detections so PPO has a numerical signal of "where is
the nearest enemy" instead of having to learn that from a 84x84 frame.

A pre-trained YOLOv8 model fine-tuned on CS2 frames is *not* shipped with this
repository. The default ``yolov8n.pt`` is an ImageNet/COCO model and will *not*
recognise CS2 enemies — it is a placeholder while you collect a labelled
dataset, fine-tune one, or download a community model (see ``docs/setup.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from cs2_rl_bot.observation.state import Team
from cs2_rl_bot.utils.config import VisionConfig
from cs2_rl_bot.utils.logging import logger

# Each detection slot exposes 8 floats to the policy:
#   valid, dx_centred, dy_centred, w_norm, h_norm, is_head, is_enemy, confidence
ENEMY_FEATURE_DIM = 8


# Class-name -> (team, is_head). Names are matched case-insensitively.
# Covers the schemas of the Vombit Hugging Face models (c/ch/t/th) and the
# more verbose Roboflow Universe datasets.
_CLASS_MAP: dict[str, tuple[str, bool]] = {
    # short codes (Vombit yolov8*_cs2)
    "c": ("CT", False),
    "ch": ("CT", True),
    "t": ("T", False),
    "th": ("T", True),
    # underscore variants
    "ct": ("CT", False),
    "ct_head": ("CT", True),
    "t_head": ("T", True),
    "head_ct": ("CT", True),
    "head_t": ("T", True),
    # body / head spelled out
    "ctbody": ("CT", False),
    "cthead": ("CT", True),
    "tbody": ("T", False),
    "thead": ("T", True),
    "ct_body": ("CT", False),
    "t_body": ("T", False),
    # had ct / had t (typo'd label seen on at least one Roboflow dataset)
    "had ct": ("CT", True),
    "had t": ("T", True),
}


def classify_class_name(name: str) -> tuple[Literal["CT", "T"] | None, bool]:
    """Map a YOLO class name to (team, is_head).

    Returns ``(None, False)`` if the class is unknown so the caller can drop
    or treat it as a non-enemy. The lookup is case-insensitive and tolerant
    of leading/trailing whitespace.
    """
    key = name.strip().lower()
    if key in _CLASS_MAP:
        return _CLASS_MAP[key]  # type: ignore[return-value]
    # Heuristic fallback: anything starting with "ct" or "c " is CT, "t" is T.
    is_head = "head" in key or key.endswith("h")
    if key.startswith("ct"):
        return ("CT", is_head)
    if key.startswith("t") and not key.startswith("ct"):
        return ("T", is_head)
    if key.startswith("c"):
        return ("CT", is_head)
    return (None, False)


@dataclass(slots=True)
class Detection:
    cls: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    team: Literal["CT", "T"] | None = None
    is_head: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "cls": self.cls,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "team": self.team,
            "is_head": self.is_head,
        }

    @property
    def centre(self) -> tuple[float, float]:
        return (
            0.5 * (self.bbox[0] + self.bbox[2]),
            0.5 * (self.bbox[1] + self.bbox[3]),
        )


class EnemyDetector:
    """Wrapper around an ultralytics YOLOv8 model.

    The model is loaded lazily so that the rest of the application can be
    imported and tested without requiring the heavy ultralytics dependency.
    """

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._model: Any | None = None

    def _maybe_load(self) -> None:
        if not self._config.enabled or self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            logger.warning(
                "ultralytics not installed — vision disabled. "
                "Install with `pip install -e '.[vision]'` ({})",
                exc,
            )
            self._config.enabled = False
            return

        device = self._config.device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info("Loading YOLO model {} on {}", self._config.model_path, device)
        self._model = YOLO(self._config.model_path)
        self._model.to(device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on an RGB image. Returns an empty list if disabled."""
        if not self._config.enabled:
            return []

        self._maybe_load()
        if self._model is None:
            return []

        results = self._model.predict(
            source=image,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            verbose=False,
        )
        out: list[Detection] = []
        for r in results:
            names = getattr(r, "names", {}) or {}
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item()) if hasattr(box, "cls") else -1
                conf = float(box.conf.item()) if hasattr(box, "conf") else 0.0
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_name = str(names.get(cls_id, str(cls_id)))
                team, is_head = classify_class_name(cls_name)
                out.append(
                    Detection(
                        cls=cls_name,
                        confidence=conf,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        team=team,
                        is_head=is_head,
                    )
                )
        return out


def encode_enemy_features(
    detections: list[Detection],
    *,
    image_shape: tuple[int, int],
    our_team: Team,
    max_slots: int,
) -> np.ndarray:
    """Encode detections as a fixed-size feature vector for the policy.

    ``image_shape`` is ``(H, W)`` of the image the detections were produced
    from — used to normalise positions to ``[-1, 1]`` (centred at the
    crosshair) for x/y and ``[0, 1]`` for width/height.

    Detections are sorted by distance from the screen centre (closest first)
    so the policy always sees the most actionable enemy in slot 0. Empty
    slots are zero-padded.

    The output shape is ``(max_slots * ENEMY_FEATURE_DIM,)``.
    """
    h, w = image_shape
    if h <= 0 or w <= 0:
        return np.zeros(max_slots * ENEMY_FEATURE_DIM, dtype=np.float32)

    cx, cy = 0.5 * w, 0.5 * h

    # Drop classes we can't interpret — they're noise to the policy.
    known = [d for d in detections if d.team is not None]
    known.sort(key=lambda d: (d.centre[0] - cx) ** 2 + (d.centre[1] - cy) ** 2)

    out = np.zeros(max_slots * ENEMY_FEATURE_DIM, dtype=np.float32)
    for i, det in enumerate(known[:max_slots]):
        x1, y1, x2, y2 = det.bbox
        det_cx, det_cy = det.centre
        dx = float(np.clip((det_cx - cx) / cx, -1.0, 1.0))
        dy = float(np.clip((det_cy - cy) / cy, -1.0, 1.0))
        bw = float(np.clip((x2 - x1) / w, 0.0, 1.0))
        bh = float(np.clip((y2 - y1) / h, 0.0, 1.0))
        is_enemy = (
            1.0
            if our_team in (Team.CT, Team.T) and det.team is not None and det.team != our_team.value
            else 0.0
        )
        offset = i * ENEMY_FEATURE_DIM
        out[offset + 0] = 1.0  # valid
        out[offset + 1] = dx
        out[offset + 2] = dy
        out[offset + 3] = bw
        out[offset + 4] = bh
        out[offset + 5] = 1.0 if det.is_head else 0.0
        out[offset + 6] = is_enemy
        out[offset + 7] = float(np.clip(det.confidence, 0.0, 1.0))
    return out
