"""Comic Text Detector teacher for manga text-mask distillation.

This module uses the ONNX export of dmMaze/zyddnys Comic Text Detector through
OpenCV DNN. By default it emits multi-point contours from the model's text
segmentation mask, so YOLO-seg labels follow the shape of the glyphs instead of
large rotated textline rectangles.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

from .paddleocr_teacher import TextRegion

logger = logging.getLogger(__name__)

CTD_MODEL_URL = (
	"https://github.com/zyddnys/manga-image-translator/releases/download/"
	"beta-0.3/comictextdetector.pt.onnx"
)
CTD_MODEL_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
CTD_MODEL_FILENAME = "comictextdetector.pt.onnx"


class ComicTextDetectorTeacher:
	"""Manga-specific text detector backed by the CTD ONNX model."""

	def __init__(
		self,
		*,
		model_path: str | Path | None = None,
		model_url: str | None = None,
		input_size: int = 1024,
		polygon_mode: str = "mask",
		mask_thresh: float = 0.3,
		box_thresh: float = 0.4,
		unclip_ratio: float = 1.5,
		contour_epsilon_ratio: float = 0.002,
		max_candidates: int = 1000,
	) -> None:
		import cv2

		if polygon_mode not in {"mask", "line_box"}:
			raise ValueError("ctd polygon_mode must be 'mask' or 'line_box'")
		self.input_size = input_size
		self.polygon_mode = polygon_mode
		self.mask_thresh = mask_thresh
		self.box_thresh = box_thresh
		self.unclip_ratio = unclip_ratio
		self.contour_epsilon_ratio = contour_epsilon_ratio
		self.max_candidates = max_candidates
		self._cv2 = cv2
		self.dedupe_priority = 100 if polygon_mode == "mask" else 20

		resolved_model_path = _resolve_model_path(model_path)
		_resolve_or_download_model(resolved_model_path, model_url or CTD_MODEL_URL)

		self._net = cv2.dnn.readNetFromONNX(str(resolved_model_path))
		self._output_layer_names = self._net.getUnconnectedOutLayersNames()
		self.device_description = f"CPU (OpenCV DNN / CTD ONNX / {polygon_mode})"

	def detect(self, image_bgr) -> list[TextRegion]:
		"""Detect text regions and return polygons in image pixels."""
		import numpy as np

		image_height, image_width = image_bgr.shape[:2]
		input_image, pad_width, pad_height = self._preprocess_image(image_bgr)
		mask_map, lines_map = self._forward_maps(input_image)
		if self.polygon_mode == "mask":
			if mask_map is None:
				raise RuntimeError("CTD ONNX did not return a text segmentation mask output.")
			polygon_candidates = self._contours_from_mask_prediction(
				self._crop_padded_map(mask_map, pad_width, pad_height),
				image_width,
				image_height,
			)
		else:
			if lines_map is None:
				raise RuntimeError("CTD ONNX did not return a textline map output.")
			polygon_candidates = self._line_boxes_from_prediction(
				self._crop_padded_map(lines_map, pad_width, pad_height),
				image_width,
				image_height,
			)

		text_regions: list[TextRegion] = []
		for polygon_points in polygon_candidates:
			polygon_array = np.asarray(polygon_points, dtype=np.float32).reshape(-1, 2)
			if polygon_array.shape[0] >= 3:
				text_regions.append(
					TextRegion(
						polygon_points=polygon_array,
						source_name="ctd",
						dedupe_priority=self.dedupe_priority,
					)
				)
		return text_regions

	def _preprocess_image(self, image_bgr):
		"""Convert BGR->RGB and letterbox to CTD's square input size."""
		cv2 = self._cv2

		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		return _letterbox(image_rgb, new_shape=(self.input_size, self.input_size), cv2_module=cv2)

	def _forward_maps(self, input_image):
		"""Run ONNX inference and select text-mask plus DB textline outputs."""
		cv2 = self._cv2

		blob = cv2.dnn.blobFromImage(
			input_image,
			scalefactor=1.0 / 255.0,
			size=(self.input_size, self.input_size),
		)
		self._net.setInput(blob)
		outputs = self._net.forward(self._output_layer_names)

		mask_map = None
		lines_map = None
		for output in outputs:
			if getattr(output, "ndim", 0) != 4:
				continue
			if output.shape[1] == 1:
				mask_map = output
			elif output.shape[1] == 2:
				# The DB textline map has two channels: shrink map and threshold map.
				lines_map = output

		return mask_map, lines_map

	def _crop_padded_map(self, lines_map, pad_width: int, pad_height: int):
		"""Remove right/bottom letterbox padding from a network output map."""
		map_height, map_width = lines_map.shape[2], lines_map.shape[3]
		crop_width = max(1, map_width - round(pad_width * map_width / self.input_size))
		crop_height = max(1, map_height - round(pad_height * map_height / self.input_size))
		return lines_map[:, :, :crop_height, :crop_width]

	def _contours_from_mask_prediction(self, prediction, image_width: int, image_height: int) -> list:
		"""Convert CTD text segmentation masks into text-shaped contours."""
		import numpy as np

		cv2 = self._cv2
		probability_map = self._squeeze_prediction_map(prediction)
		probability_map = cv2.resize(
			probability_map,
			(image_width, image_height),
			interpolation=cv2.INTER_LINEAR,
		)
		bitmap = (probability_map > self.mask_thresh).astype(np.uint8) * 255
		contours, _ = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[: self.max_candidates]

		polygons: list = []
		for contour in contours:
			if contour.shape[0] < 3:
				continue
			arc_length = cv2.arcLength(contour, True)
			if arc_length <= 0.0:
				continue
			epsilon = max(0.0, self.contour_epsilon_ratio) * arc_length
			approximated_contour = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0.0 else contour
			polygon = approximated_contour.reshape(-1, 2).astype(np.float32)
			if polygon.shape[0] < 3:
				polygon = contour.reshape(-1, 2).astype(np.float32)
			if polygon.shape[0] >= 3:
				polygons.append(polygon)
		return polygons

	def _line_boxes_from_prediction(self, prediction, image_width: int, image_height: int) -> list:
		"""Convert CTD DB textline maps into rotated quadrilateral polygons."""
		import numpy as np

		cv2 = self._cv2
		probability_map = self._squeeze_prediction_map(prediction)
		bitmap = probability_map > self.mask_thresh
		map_height, map_width = bitmap.shape
		contours, _ = cv2.findContours(
			(bitmap.astype(np.uint8) * 255),
			cv2.RETR_LIST,
			cv2.CHAIN_APPROX_SIMPLE,
		)

		polygons: list = []
		for contour in contours[: self.max_candidates]:
			contour_points = contour.squeeze(1)
			if contour_points.ndim != 2 or contour_points.shape[0] < 3:
				continue

			mini_box, short_side = self._get_mini_box(contour_points)
			if short_side < 2:
				continue

			score = self._box_score_fast(probability_map, contour_points)
			if score <= self.box_thresh:
				continue

			expanded_box = self._unclip(np.asarray(mini_box, dtype=np.float32))
			if expanded_box.size == 0:
				continue

			expanded_contour = expanded_box.reshape(-1, 1, 2).astype(np.float32)
			quad_points, expanded_short_side = self._get_mini_box(expanded_contour)
			if expanded_short_side < 2:
				continue

			polygon = np.asarray(quad_points, dtype=np.float32)
			polygon[:, 0] = np.clip(np.round(polygon[:, 0] / map_width * image_width), 0, image_width)
			polygon[:, 1] = np.clip(np.round(polygon[:, 1] / map_height * image_height), 0, image_height)
			polygons.append(polygon)

		return polygons

	@staticmethod
	def _squeeze_prediction_map(prediction):
		"""Return the first probability channel as a 2D float map."""
		if prediction.ndim == 4:
			return prediction[0, 0, :, :]
		if prediction.ndim == 3:
			return prediction[0, :, :]
		return prediction.squeeze()

	def _unclip(self, box):
		"""Expand a small DB box to recover the full textline extent."""
		import numpy as np
		import pyclipper

		cv2 = self._cv2
		area = abs(cv2.contourArea(box.astype(np.float32)))
		perimeter = cv2.arcLength(box.reshape(-1, 1, 2).astype(np.float32), True)
		if area <= 0.0 or perimeter <= 0.0:
			return np.empty((0, 2), dtype=np.float32)

		distance = area * self.unclip_ratio / perimeter
		path = [tuple(point) for point in np.round(box).astype(int).tolist()]
		offset = pyclipper.PyclipperOffset()
		offset.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
		expanded_paths = offset.Execute(distance)
		if not expanded_paths:
			return np.empty((0, 2), dtype=np.float32)

		# pyclipper may return multiple paths for degenerate shapes; keep the largest.
		largest_path = max(
			expanded_paths,
			key=lambda candidate: abs(cv2.contourArea(np.asarray(candidate, dtype=np.float32))),
		)
		return np.asarray(largest_path, dtype=np.float32).reshape(-1, 2)

	def _get_mini_box(self, contour):
		"""Return CTD/DBNet's canonical four-point rotated rectangle order."""
		import numpy as np

		cv2 = self._cv2

		bounding_box = cv2.minAreaRect(contour.astype(np.float32))
		points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda point: point[0])

		if points[1][1] > points[0][1]:
			index_1, index_4 = 0, 1
		else:
			index_1, index_4 = 1, 0
		if points[3][1] > points[2][1]:
			index_2, index_3 = 2, 3
		else:
			index_2, index_3 = 3, 2

		return [points[index_1], points[index_2], points[index_3], points[index_4]], min(bounding_box[1])

	def _box_score_fast(self, probability_map, box) -> float:
		"""Mean DB probability inside a contour's bounding window."""
		import numpy as np

		cv2 = self._cv2
		map_height, map_width = probability_map.shape[:2]
		scored_box = box.copy()
		x_min = np.clip(np.floor(scored_box[:, 0].min()).astype(np.int32), 0, map_width - 1)
		x_max = np.clip(np.ceil(scored_box[:, 0].max()).astype(np.int32), 0, map_width - 1)
		y_min = np.clip(np.floor(scored_box[:, 1].min()).astype(np.int32), 0, map_height - 1)
		y_max = np.clip(np.ceil(scored_box[:, 1].max()).astype(np.int32), 0, map_height - 1)
		if x_max < x_min or y_max < y_min:
			return 0.0

		mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
		scored_box[:, 0] -= x_min
		scored_box[:, 1] -= y_min
		cv2.fillPoly(mask, scored_box.reshape(1, -1, 2).astype(np.int32), 1)
		return float(cv2.mean(probability_map[y_min : y_max + 1, x_min : x_max + 1], mask)[0])


def _letterbox(image, *, new_shape: tuple[int, int], cv2_module):
	"""Resize image with unchanged aspect ratio and bottom/right padding."""
	image_height, image_width = image.shape[:2]
	resize_ratio = min(new_shape[0] / image_height, new_shape[1] / image_width)
	unpadded_width = int(round(image_width * resize_ratio))
	unpadded_height = int(round(image_height * resize_ratio))
	pad_width = new_shape[1] - unpadded_width
	pad_height = new_shape[0] - unpadded_height

	if (image_width, image_height) != (unpadded_width, unpadded_height):
		image = cv2_module.resize(image, (unpadded_width, unpadded_height), interpolation=cv2_module.INTER_LINEAR)
	image = cv2_module.copyMakeBorder(
		image,
		0,
		pad_height,
		0,
		pad_width,
		cv2_module.BORDER_CONSTANT,
		value=(0, 0, 0),
	)
	return image, pad_width, pad_height


def _resolve_model_path(model_path: str | Path | None) -> Path:
	if model_path is not None:
		return Path(model_path).expanduser()

	from core import paths

	return Path(paths.models_dir("ctd", CTD_MODEL_FILENAME))


def _resolve_or_download_model(model_path: Path, model_url: str) -> None:
	if model_path.is_file() and (model_url != CTD_MODEL_URL or _sha256(model_path) == CTD_MODEL_SHA256):
		return

	if model_path.is_file():
		logger.warning("CTD model hash mismatch at %s; downloading a fresh copy.", model_path)
	else:
		logger.info("Downloading CTD model to %s", model_path)

	model_path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")
	try:
		_download_file(model_url, temporary_path)
		if model_url == CTD_MODEL_URL:
			digest = _sha256(temporary_path)
			if digest != CTD_MODEL_SHA256:
				raise RuntimeError(
					"Downloaded CTD model hash mismatch: "
					f"expected {CTD_MODEL_SHA256}, got {digest}"
				)
		temporary_path.replace(model_path)
	finally:
		if temporary_path.exists():
			temporary_path.unlink()


def _download_file(url: str, destination_path: Path) -> None:
	with urllib.request.urlopen(url) as response, destination_path.open("wb") as output_file:
		while True:
			chunk = response.read(1024 * 1024)
			if not chunk:
				break
			output_file.write(chunk)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as input_file:
		while True:
			chunk = input_file.read(1024 * 1024)
			if not chunk:
				break
			digest.update(chunk)
	return digest.hexdigest()
