"""Website-style prediction overlays for segmentation results.

Renders the same "demo" look prediction sites use: each instance's mask is
painted with a per-class colour (translucent fill + solid outline) and tagged
with its class name on top. Backend-agnostic: it only needs an instance's
``label`` and ``mask``, so it works for any :class:`~core.segmenter.Instance`.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Iterable, Mapping

import cv2 as cv
import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle (segmenter imports this module)
	from core.segmenter import Instance

# Per-class colours for the manga-segment dataset, as web hex (#RRGGBB).
DEFAULT_CLASS_COLORS: dict[str, str] = {
	"comic": "#C7FC00",
	"caption-box": "#00B7EB",
	"speech-balloon": "#8622FF",
	"text": "#FE0056",
	"thought-balloon": "#FF8000",
}


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
	"""Convert a ``#RRGGBB`` (or ``RRGGBB``) hex colour to an OpenCV BGR tuple."""
	value = value.lstrip("#")
	r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
	return (b, g, r)


def _resolve_color(label: str, colors: Mapping[str, str]) -> tuple[int, int, int]:
	"""BGR colour for a class: the configured one, else a deterministic fallback."""
	if label in colors:
		return _hex_to_bgr(colors[label])
	digest = hashlib.md5(label.encode("utf-8")).digest()
	return (int(digest[0]), int(digest[1]), int(digest[2]))


def _text_color_for(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
	"""Black or white text, whichever contrasts better with ``bgr``."""
	b, g, r = bgr
	luminance = 0.114 * b + 0.587 * g + 0.299 * r
	return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _binarize(mask: np.ndarray, width: int, height: int) -> np.ndarray | None:
	"""Binarise a (H, W) float/int mask to 0/1 ``uint8``, resized to the image."""
	binary = np.where(mask > 0.5, 1, 0).astype(np.uint8)
	if binary.shape != (height, width):
		binary = cv.resize(binary, (width, height), interpolation=cv.INTER_NEAREST)
	if not binary.any():
		return None
	return binary


def _draw_label(
	canvas: np.ndarray,
	mask: np.ndarray,
	label: str,
	color: tuple[int, int, int],
	*,
	scale: float,
	thickness: int,
) -> None:
	"""Draw a filled name tag at the top-left of ``mask``'s bounding box."""
	ys, xs = np.where(mask > 0)
	if len(xs) == 0:
		return
	x0, y0 = int(xs.min()), int(ys.min())

	font = cv.FONT_HERSHEY_SIMPLEX
	(text_w, text_h), baseline = cv.getTextSize(label, font, scale, thickness)
	pad = max(2, int(round(scale * 4)))
	box_w, box_h = text_w + 2 * pad, text_h + baseline + 2 * pad

	height, width = canvas.shape[:2]
	# Sit the tag just above the object; drop it inside when there is no room.
	top = y0 - box_h if y0 - box_h >= 0 else y0
	left = min(max(x0, 0), max(width - box_w, 0))
	top = min(max(top, 0), max(height - box_h, 0))

	cv.rectangle(canvas, (left, top), (left + box_w, top + box_h), color, -1)
	cv.putText(
		canvas,
		label,
		(left + pad, top + pad + text_h),
		font,
		scale,
		_text_color_for(color),
		thickness,
		cv.LINE_AA,
	)


def draw_class_overlay(
	image_bgr: np.ndarray,
	instances: Iterable["Instance"],
	*,
	colors: Mapping[str, str] | None = None,
	alpha: float = 0.4,
) -> np.ndarray:
	"""Return ``image_bgr`` with each instance drawn as a coloured, labelled mask.

	Masks are painted as a translucent fill first, then their outlines and class
	name tags are drawn on the blended canvas so they stay crisp. Colours come
	from ``colors`` (class name -> hex), defaulting to :data:`DEFAULT_CLASS_COLORS`
	with a deterministic per-name fallback for unknown classes.
	"""
	colors = colors or DEFAULT_CLASS_COLORS
	height, width = image_bgr.shape[:2]
	canvas = image_bgr.copy()
	fill = image_bgr.copy()

	# Sizes scale with the page so tags/outlines stay legible on large scans.
	scale = max(0.5, min(1.5, width / 1600))
	line_thickness = max(2, int(round(width / 600)))
	text_thickness = max(1, int(round(scale * 2)))

	drawn: list[tuple[str, np.ndarray, tuple[int, int, int]]] = []
	for instance in instances:
		mask = _binarize(instance.mask, width, height)
		if mask is None:
			continue
		color = _resolve_color(instance.label, colors)
		fill[mask > 0] = color
		drawn.append((instance.label, mask, color))

	cv.addWeighted(fill, alpha, canvas, 1.0 - alpha, 0.0, canvas)

	for label, mask, color in drawn:
		contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		cv.drawContours(canvas, contours, -1, color, line_thickness)
		_draw_label(
			canvas, mask, label, color, scale=scale, thickness=text_thickness
		)

	return canvas
