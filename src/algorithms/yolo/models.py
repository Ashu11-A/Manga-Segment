"""YOLO model-family catalogue.

Single source of truth for every supported base model. Used by the CLI
(``--base-model`` choices) and the interactive wizard (two-step family → size
selection).
"""
from __future__ import annotations

FAMILIES: dict[str, list[str]] = {
    "YOLOv8-seg":  ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
    "YOLOv9-seg":  ["yolov9t-seg.pt", "yolov9s-seg.pt", "yolov9m-seg.pt", "yolov9c-seg.pt", "yolov9e-seg.pt"],
    "YOLOv10-seg": ["yolov10n-seg.pt", "yolov10s-seg.pt", "yolov10m-seg.pt", "yolov10b-seg.pt", "yolov10l-seg.pt", "yolov10x-seg.pt"],
    "YOLO11-seg":  ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"],
    "YOLO12-seg":  ["yolo12n-seg.pt", "yolo12s-seg.pt", "yolo12m-seg.pt", "yolo12l-seg.pt", "yolo12x-seg.pt"],
    "YOLO26-seg":  ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt"],
}

ALL_MODELS: list[str] = [m for models in FAMILIES.values() for m in models]

DEFAULT_MODEL = "yolo26s-seg.pt"


def family_of(model_name: str) -> str | None:
    """Return the family key that owns *model_name*, or ``None``."""
    for family, models in FAMILIES.items():
        if model_name in models:
            return family
    return None
