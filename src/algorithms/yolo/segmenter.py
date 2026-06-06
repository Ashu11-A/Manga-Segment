"""YOLO segmentation inference as a :class:`~core.segmenter.BaseSegmenter`.

Loads a trained Ultralytics segmentation checkpoint and turns images into
:class:`~core.segmenter.SegmentationResult` instances. The single-image
:meth:`YoloSegmenter.predict` powers the serving path and the base directory
workflow, while :meth:`YoloSegmenter.segment_directory` is overridden to keep
the efficient Ultralytics streamed-source handling used for batch inference.
"""

from __future__ import annotations

import os

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from core.device import resolve_device, supports_half
from core.segmenter import BaseSegmenter, Instance, SegmentationResult


class YoloSegmenter(BaseSegmenter):
	"""Run segmentation inference and composite background-removed PNGs."""

	def __init__(
		self,
		weights: str,
		*,
		device: str | None = None,
		imgsz: int | list[int] = 1280,
		conf: float = 0.5,
		iou: float = 0.7,
		max_det: int = 300,
		retina_masks: bool = True,
		agnostic_nms: bool = True,
		augment: bool = False,
		classes: list[int] | None = None,
		verbose: bool = False,
	) -> None:
		self.device = resolve_device(device)
		# Half precision is only valid on CUDA; enabling it on CPU raises. It is
		# also incompatible with retina_masks: Ultralytics' process_mask_native
		# multiplies half mask coefficients by float protos, raising a dtype
		# mismatch, so FP16 inference is only kept when retina_masks is off.
		self.half = supports_half(self.device) and not retina_masks
		self.imgsz = imgsz
		self.conf = conf
		self.iou = iou
		self.max_det = max_det
		self.retina_masks = retina_masks
		self.agnostic_nms = agnostic_nms
		self.augment = augment
		self.classes = classes
		self.verbose = verbose

		self.model = YOLO(weights, task="segment")
		print(f"📦 Loaded segmentation model: {weights} (device={self.device}, half={self.half})")

	# -- prediction kwargs ---------------------------------------------------
	def _predict_kwargs(self) -> dict:
		return {
			"device": self.device,
			"half": self.half,
			"imgsz": self.imgsz,
			"conf": self.conf,
			"iou": self.iou,
			"max_det": self.max_det,
			"retina_masks": self.retina_masks,
			"agnostic_nms": self.agnostic_nms,
			"augment": self.augment,
			"classes": self.classes,
			"verbose": self.verbose,
		}

	# -- single-image prediction --------------------------------------------
	def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
		"""Segment a single BGR image into a :class:`SegmentationResult`."""
		results = self.model.predict(source=image_bgr, stream=False, **self._predict_kwargs())
		result = results[0]
		return self._to_result(result, image_bgr=result.orig_img)

	# -- result conversion ---------------------------------------------------
	@staticmethod
	def _to_result(result: Results, *, image_bgr: np.ndarray, stem: str = "") -> SegmentationResult:
		"""Build a :class:`SegmentationResult` from one Ultralytics ``Results``."""
		if result.masks is None or result.boxes is None or len(result.masks) == 0:
			return SegmentationResult(image_bgr=image_bgr, instances=[], stem=stem)

		masks = result.masks.data.cpu().numpy()  # (N, H, W), float in [0, 1]
		class_ids = result.boxes.cls.cpu().numpy().astype(int)
		names = result.names

		instances = [
			Instance(label=names[int(class_ids[index])], mask=mask)
			for index, mask in enumerate(masks)
		]
		return SegmentationResult(image_bgr=image_bgr, instances=instances, stem=stem)

	# -- folder workflow (streamed) -----------------------------------------
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
		"""Segment every image in ``images_dir`` and write outputs to ``output_dir``.

		Overrides the base loop to keep Ultralytics' efficient streamed source
		handling: feeding the directory (not a list of paths) lets Ultralytics
		preserve each original filename on ``result.path``, which names outputs.
		Returns the list of written files.
		"""
		os.makedirs(output_dir, exist_ok=True)

		written: list[str] = []
		count = 0
		# Feed the directory (not a list of paths): Ultralytics then preserves each
		# original filename on ``result.path``, which we use to name the outputs.
		for result in self.model.predict(source=images_dir, stream=True, **self._predict_kwargs()):
			count += 1
			stem = os.path.splitext(os.path.basename(result.path))[0]
			seg_result = self._to_result(result, image_bgr=result.orig_img, stem=stem)

			written.extend(
				self._write_result(
					seg_result,
					output_dir,
					keep_classes=keep_classes,
					ignore_classes=ignore_classes,
					save_masks=save_masks,
					save_segmented=save_segmented,
					draw_overlay=draw_overlay,
				)
			)

		print(f"✅ Done: {count} image(s) processed, {len(written)} file(s) written to {output_dir}")
		return written

	# -- annotated visualisation --------------------------------------------
	def annotate(self, image_bgr: np.ndarray) -> np.ndarray:
		"""Return the BGR image with detections drawn on it (serving "annotated")."""
		results = self.model.predict(source=image_bgr, stream=False, **self._predict_kwargs())
		return results[0].plot()
