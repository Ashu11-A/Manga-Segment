"""U-Net segmentation inference as a :class:`~core.segmenter.BaseSegmenter`.

Loads the pre-trained ``SavedModel`` from ``models/unet`` and turns each BGR
image into a :class:`~core.segmenter.SegmentationResult`: a single
``foreground`` instance (thresholded max over the 4 predicted channels) plus the
raw predicted RGBA mask as an extra artifact. This reproduces the legacy
``UnetModel.test`` outputs (``<stem>_unet_mask.png`` + ``<stem>_segmented.png``)
while reusing the shared compositing in :mod:`core.imaging`.

TensorFlow is imported lazily inside :meth:`UnetSegmenter._load` so that
``import algorithms`` stays cheap.
"""

from __future__ import annotations

import os

import cv2 as cv
import numpy as np

from core import paths
from core.segmenter import BaseSegmenter, Instance, SegmentationResult


class UnetSegmenter(BaseSegmenter):
	"""Run U-Net inference and emit the predicted RGBA mask + foreground composite."""

	def __init__(self, model_dir: str | None = None, *, threshold: float = 0.5) -> None:
		self.model_dir = model_dir if model_dir is not None else paths.models_dir("unet")
		self.threshold = threshold
		# Default input resolution; overwritten by the real SavedModel signature.
		self.input_height = 768
		self.input_width = 512
		self._model = None
		self._predict_fn = None
		self._input_key = "input_1"

	def _load(self) -> None:
		"""Load the SavedModel once (lazy loading)."""
		if self._predict_fn is not None:
			return

		import tensorflow as tf

		if not os.path.isdir(self.model_dir):
			raise FileNotFoundError(
				f"⚠️ U-Net model not found at: {self.model_dir}"
			)

		print(f"📦 Loading U-Net model from: {self.model_dir}")
		self._model = tf.saved_model.load(self.model_dir)

		if "serving_default" not in self._model.signatures:
			raise ValueError(
				"⚠️ The SavedModel has no 'serving_default' signature."
			)

		self._predict_fn = self._model.signatures["serving_default"]

		# Discover the input name and resolution from the signature.
		input_spec = self._predict_fn.structured_input_signature[1]
		self._input_key = next(iter(input_spec))
		shape = input_spec[self._input_key].shape

		if shape.rank == 4:
			if shape[1] is not None:
				self.input_height = int(shape[1])
			if shape[2] is not None:
				self.input_width = int(shape[2])

		print(
			f"✅ Model loaded. Input '{self._input_key}': "
			f"{self.input_height}x{self.input_width} (RGBA)"
		)

	def _preprocess(self, image_data: np.ndarray):
		"""Convert a BGR/BGRA (OpenCV) image into the tensor the model expects."""
		import tensorflow as tf

		channels = image_data.shape[2] if image_data.ndim == 3 else 1

		if channels == 4:
			rgba = cv.cvtColor(image_data, cv.COLOR_BGRA2RGBA)
		elif channels == 3:
			rgba = cv.cvtColor(image_data, cv.COLOR_BGR2RGBA)
		else:
			rgba = cv.cvtColor(image_data, cv.COLOR_GRAY2RGBA)

		# OpenCV uses (width, height) in resize.
		resized = cv.resize(
			rgba,
			(self.input_width, self.input_height),
			interpolation=cv.INTER_NEAREST,
		)
		normalized = resized.astype(np.float32) / 255.0
		return tf.convert_to_tensor(normalized[np.newaxis, ...])

	def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
		"""Segment a single BGR image into a :class:`SegmentationResult`.

		Mirrors the legacy ``UnetModel.test``: the predicted 4-channel mask in
		``[0, 1]`` becomes a resized RGBA ``unet_mask`` extra, and a thresholded
		max-over-channels foreground becomes the single ``foreground`` instance
		(whose composite reproduces the legacy ``<stem>_segmented.png``).
		"""
		self._load()
		assert self._predict_fn is not None

		input_tensor = self._preprocess(image_bgr)
		outputs = self._predict_fn(**{self._input_key: input_tensor})
		prediction = next(iter(outputs.values())).numpy()[0]

		height, width = image_bgr.shape[:2]

		# 1) Predicted RGBA mask, resized back to the original resolution.
		mask_rgba = (np.clip(prediction, 0.0, 1.0) * 255).astype(np.uint8)
		mask_bgra = cv.cvtColor(mask_rgba, cv.COLOR_RGBA2BGRA)
		mask_bgra = cv.resize(mask_bgra, (width, height))

		# 2) Foreground: max over channels, resized, thresholded to a binary mask.
		foreground = np.max(prediction, axis=-1)
		foreground = cv.resize(foreground, (width, height))
		foreground_mask = (foreground >= self.threshold).astype(np.uint8) * 255

		return SegmentationResult(
			image_bgr=image_bgr,
			instances=[Instance(label="foreground", mask=foreground_mask)],
			extras={"unet_mask": mask_bgra},
		)
