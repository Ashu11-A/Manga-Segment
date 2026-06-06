"""Segmentation result types and the shared inference workflow.

``BaseSegmenter`` owns the parts that every segmentation backend repeats: the
per-directory loop, output-file naming, per-instance mask writing and the
background-removed PNG compositing. A concrete backend only has to implement
:meth:`BaseSegmenter.predict` — turning one BGR image into a
:class:`SegmentationResult` — and it gets the directory workflow, single-array
helper (for serving) and consistent outputs for free.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from core.imaging import (
	combine_masks,
	composite_rgba,
	iter_image_files,
	read_image_bgr,
	save_mask,
)
from core.visualize import draw_class_overlay


@dataclass
class Instance:
	"""One detected/segmented instance: a class label and its (H, W) mask."""

	label: str
	mask: np.ndarray


@dataclass
class SegmentationResult:
	"""Backend-agnostic output of a single image's segmentation.

	``extras`` maps an output-name suffix to a ready-to-write image (e.g.
	``{"unet_mask": bgra}`` -> ``<stem>_unet_mask.png``), letting a backend emit
	extra artifacts without special-casing the writer.
	"""

	image_bgr: np.ndarray
	instances: list[Instance]
	stem: str = ""
	extras: dict[str, np.ndarray] = field(default_factory=dict)


class BaseSegmenter(ABC):
	"""Template for segmentation inference; backends implement :meth:`predict`."""

	@abstractmethod
	def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
		"""Segment a single BGR image into a :class:`SegmentationResult`."""

	# -- class filtering -----------------------------------------------------
	@staticmethod
	def _is_kept(
		label: str,
		keep_classes: list[str] | None,
		ignore_classes: list[str] | None,
	) -> bool:
		"""Whether an instance's class contributes to the composited foreground.

		``keep_classes`` is an allow-list (``None`` keeps all), ``ignore_classes``
		a deny-list applied on top of it (``None`` ignores none). An instance is
		kept only when it is allowed and not ignored.
		"""
		if keep_classes is not None and label not in keep_classes:
			return False
		if ignore_classes is not None and label in ignore_classes:
			return False
		return True

	# -- directory workflow --------------------------------------------------
	def segment_directory(
		self,
		images_dir: str,
		output_dir: str,
		*,
		keep_classes: list[str] | None = None,
		ignore_classes: list[str] | None = None,
		save_masks: bool = True,
		save_segmented: bool = True,
		draw_overlay: bool = False,
	) -> list[str]:
		"""Segment every image in ``images_dir``; write outputs to ``output_dir``.

		``keep_classes`` optionally restricts which classes contribute to the
		composited foreground (default: all); ``ignore_classes`` drops classes
		from it (e.g. ``["text"]``). ``draw_overlay`` additionally writes a
		website-style ``<stem>_overlay.png`` (coloured, labelled masks). Returns
		the written file paths. Backends with batched/streamed prediction may
		override this while reusing :meth:`_write_result`.
		"""
		names = iter_image_files(images_dir)
		if not names:
			print(f"⚠️ No images found in {images_dir}")
			return []

		os.makedirs(output_dir, exist_ok=True)

		written: list[str] = []
		for name in names:
			image = read_image_bgr(os.path.join(images_dir, name))
			if image is None:
				print(f"⚠️ Could not read image: {name}")
				continue
			result = self.predict(image)
			result.stem = os.path.splitext(name)[0]
			written.extend(
				self._write_result(
					result,
					output_dir,
					keep_classes=keep_classes,
					ignore_classes=ignore_classes,
					save_masks=save_masks,
					save_segmented=save_segmented,
					draw_overlay=draw_overlay,
				)
			)

		print(
			f"✅ Done: {len(names)} image(s) processed, "
			f"{len(written)} file(s) written to {output_dir}"
		)
		return written

	# -- single-image helper (used by serving) -------------------------------
	def segment_array(
		self,
		image_bgr: np.ndarray,
		*,
		keep_classes: list[str] | None = None,
		ignore_classes: list[str] | None = None,
	) -> np.ndarray | None:
		"""Segment one BGR image and return a BGRA composite (or ``None``).

		``None`` means nothing was detected, so callers can fall back to the
		original image.
		"""
		result = self.predict(image_bgr)
		selected = [
			inst.mask
			for inst in result.instances
			if self._is_kept(inst.label, keep_classes, ignore_classes)
		]
		combined = combine_masks(selected)
		if combined is None:
			return None
		return composite_rgba(image_bgr, combined)

	# -- shared output writer ------------------------------------------------
	def _write_result(
		self,
		result: SegmentationResult,
		output_dir: str,
		*,
		keep_classes: list[str] | None,
		ignore_classes: list[str] | None = None,
		save_masks: bool,
		save_segmented: bool,
		draw_overlay: bool = False,
	) -> list[str]:
		"""Write per-instance masks, extras and the composited PNG for one result."""
		stem = result.stem
		written: list[str] = []

		# Backend-specific extra artifacts (e.g. the U-Net raw RGBA mask).
		for suffix, image in result.extras.items():
			path = os.path.join(output_dir, f"{stem}_{suffix}.png")
			cv.imwrite(path, image)
			written.append(path)

		if not result.instances:
			print(f"⚠️ No segments detected: {stem}")
			return written

		# Website-style prediction overlay: coloured, labelled masks for the kept
		# classes drawn on the original image.
		if draw_overlay:
			drawn = [
				instance
				for instance in result.instances
				if self._is_kept(instance.label, keep_classes, ignore_classes)
			]
			overlay = draw_class_overlay(result.image_bgr, drawn)
			overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
			cv.imwrite(overlay_path, overlay)
			written.append(overlay_path)

		selected: list[np.ndarray] = []
		for index, instance in enumerate(result.instances):
			if save_masks:
				mask_path = os.path.join(
					output_dir, f"{stem}_{instance.label}_{index}_mask.png"
				)
				save_mask(mask_path, instance.mask)
				written.append(mask_path)
			if self._is_kept(instance.label, keep_classes, ignore_classes):
				selected.append(instance.mask)

		if save_segmented:
			combined = combine_masks(selected)
			if combined is not None:
				segmented = composite_rgba(result.image_bgr, combined)
				seg_path = os.path.join(output_dir, f"{stem}_segmented.png")
				cv.imwrite(seg_path, segmented)
				written.append(seg_path)

		print(f"🖼️  {stem}: {len(result.instances)} segment(s)")
		return written
