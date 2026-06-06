"""Image IO and mask compositing shared by every segmentation algorithm.

This is the single home for the logic that used to be copied between the YOLO
inference pipeline (``composite_rgba``) and the U-Net model (``_apply_mask``),
plus the image-extension lists scattered across the old modules.
"""

from __future__ import annotations

import os

import cv2 as cv
import numpy as np

# Recognised raster inputs for directory-based inference.
IMAGE_EXTS: tuple[str, ...] = (
	".png",
	".jpg",
	".jpeg",
	".bmp",
	".webp",
	".tif",
	".tiff",
)


def iter_image_files(directory: str) -> list[str]:
	"""Sorted list of image *file names* (not paths) inside ``directory``.

	Names are returned (rather than full paths) so callers can derive output
	stems with :func:`os.path.splitext` directly, matching the legacy behaviour.
	"""
	if not os.path.isdir(directory):
		raise FileNotFoundError(f"⚠️ Images directory not found: {directory}")
	return sorted(
		name
		for name in os.listdir(directory)
		if name.lower().endswith(IMAGE_EXTS)
	)


def read_image_bgr(path: str) -> np.ndarray | None:
	"""Read an image as BGR (OpenCV convention); ``None`` when unreadable."""
	return cv.imread(path, cv.IMREAD_COLOR)


def save_mask(path: str, mask: np.ndarray) -> None:
	"""Write a single-channel mask as 8-bit grayscale PNG.

	Accepts either a float mask in ``[0, 1]`` or an already-scaled ``uint8``
	mask and normalises to ``uint8`` before writing.
	"""
	if mask.dtype != np.uint8:
		mask = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
	cv.imwrite(path, mask)


def combine_masks(masks: list[np.ndarray]) -> np.ndarray | None:
	"""Union a list of (H, W) masks into one binary ``uint8`` mask (0/255).

	Returns ``None`` for an empty list. Masks may be float or integer; any
	positive value is treated as foreground.
	"""
	if not masks:
		return None
	stacked = np.stack(masks)
	return (np.any(stacked > 0, axis=0).astype(np.uint8)) * 255


def composite_rgba(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""Return a BGRA image keeping only the masked region (rest transparent).

	The mask is binarised, resized to the image with nearest-neighbour when its
	shape differs, used to zero out the background and written into the alpha
	channel. Single implementation shared by the YOLO and U-Net pipelines.
	"""
	binary = np.where(mask > 0, 255, 0).astype(np.uint8)

	if image_bgr.shape[:2] != binary.shape:
		binary = cv.resize(
			binary,
			(image_bgr.shape[1], image_bgr.shape[0]),
			interpolation=cv.INTER_NEAREST,
		)

	foreground = cv.bitwise_and(image_bgr, image_bgr, mask=binary)
	bgra = cv.cvtColor(foreground, cv.COLOR_BGR2BGRA)
	bgra[:, :, 3] = binary
	return bgra
