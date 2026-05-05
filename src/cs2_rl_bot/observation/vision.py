"""Optional YOLOv8-based enemy detector.

`ultralytics` is an optional dependency (install with ``pip install -e '.[vision]'``).
If it is not installed, the detector is a no-op. The detector returns axis-aligned
bounding boxes in the same coordinate frame as the input image.

A pre-trained YOLOv8 model fine-tuned on CS2 frames is *not* shipped with this
repository. The default ``yolov8n.pt`` is an ImageNet/COCO model and will *not*
recognise CS2 enemies — it is a placeholder while you collect a labelled
dataset and fine-tune. See ``docs/architecture.md`` for guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cs2_rl_bot.utils.config import VisionConfig
from cs2_rl_bot.utils.logging import logger


@dataclass(slots=True)
class Detection:
    cls: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    def as_dict(self) -> dict[str, Any]:
        return {
            "cls": self.cls,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }


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
                out.append(
                    Detection(
                        cls=str(names.get(cls_id, str(cls_id))),
                        confidence=conf,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    )
                )
        return out
