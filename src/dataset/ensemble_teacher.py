"""Teacher ensemble with polygon-level duplicate suppression."""

from __future__ import annotations

from dataclasses import dataclass

from .paddleocr_teacher import TextRegion


class EnsembleTextTeacher:
	"""Runs multiple text teachers and removes duplicated detections.

	The duplicate pass uses two related tests: polygon IoU catches same-size boxes,
	while intersection-over-smaller-area catches one model's larger textline polygon
	covering another model's smaller fragment polygon.
	"""

	def __init__(
		self,
		teachers: list[tuple[str, object]],
		*,
		iou_threshold: float = 0.5,
		overlap_threshold: float = 0.8,
	) -> None:
		if not teachers:
			raise ValueError("ensemble requires at least one teacher")
		self.teachers = teachers
		self.iou_threshold = iou_threshold
		self.overlap_threshold = overlap_threshold
		teacher_descriptions = [
			f"{teacher_name}: {teacher_model.device_description}"
			for teacher_name, teacher_model in teachers
		]
		self.device_description = "ensemble [" + "; ".join(teacher_descriptions) + "]"

	def detect(self, image_bgr) -> list[TextRegion]:
		collected_regions: list[TextRegion] = []
		for teacher_name, teacher_model in self.teachers:
			teacher_priority = getattr(teacher_model, "dedupe_priority", 10)
			for region in teacher_model.detect(image_bgr):
				if not region.source_name:
					region.source_name = teacher_name
				if not region.dedupe_priority:
					region.dedupe_priority = teacher_priority
				collected_regions.append(region)
		return deduplicate_text_regions(
			collected_regions,
			iou_threshold=self.iou_threshold,
			overlap_threshold=self.overlap_threshold,
		)


@dataclass
class _RegionCandidate:
	original_index: int
	region: TextRegion
	polygon_points: object
	shapely_polygon: object
	bounding_box: tuple[float, float, float, float]
	area: float
	source_name: str
	dedupe_priority: int
	point_count: int


def deduplicate_text_regions(
	regions: list[TextRegion],
	*,
	iou_threshold: float = 0.5,
	overlap_threshold: float = 0.8,
) -> list[TextRegion]:
	"""Remove duplicate text regions using priority-ordered polygon NMS."""
	if len(regions) <= 1:
		return regions

	try:
		import numpy as np
		from shapely.geometry import Polygon as ShapelyPolygon
	except Exception:  # pragma: no cover - dependencies are present in run.py
		return regions

	candidates: list[_RegionCandidate] = []
	for original_index, region in enumerate(regions):
		polygon_points = np.asarray(region.polygon_points, dtype=np.float32).reshape(-1, 2)
		if polygon_points.shape[0] < 3:
			continue

		shapely_polygon = ShapelyPolygon(polygon_points)
		if not shapely_polygon.is_valid:
			shapely_polygon = shapely_polygon.buffer(0)
		if shapely_polygon.is_empty or shapely_polygon.area <= 0.0:
			continue

		candidates.append(
			_RegionCandidate(
				original_index=original_index,
				region=region,
				polygon_points=polygon_points,
				shapely_polygon=shapely_polygon,
				bounding_box=(
					float(polygon_points[:, 0].min()),
					float(polygon_points[:, 1].min()),
					float(polygon_points[:, 0].max()),
					float(polygon_points[:, 1].max()),
				),
				area=float(shapely_polygon.area),
				source_name=region.source_name,
				dedupe_priority=region.dedupe_priority,
				point_count=int(polygon_points.shape[0]),
			)
		)

	if len(candidates) <= 1:
		return [candidate.region for candidate in candidates]

	ordered_candidate_indices = sorted(
		range(len(candidates)),
		key=lambda candidate_index: _candidate_preference(candidates[candidate_index]),
		reverse=True,
	)
	suppressed_indices: set[int] = set()
	kept_candidate_indices: list[int] = []
	for candidate_index in ordered_candidate_indices:
		if candidate_index in suppressed_indices:
			continue
		kept_candidate_indices.append(candidate_index)
		candidate = candidates[candidate_index]
		for other_index in ordered_candidate_indices:
			if other_index == candidate_index or other_index in suppressed_indices:
				continue
			other_candidate = candidates[other_index]
			if not _bounding_boxes_overlap(candidate.bounding_box, other_candidate.bounding_box):
				continue
			if _regions_overlap_enough(
				candidate,
				other_candidate,
				iou_threshold=iou_threshold,
				overlap_threshold=overlap_threshold,
			):
				suppressed_indices.add(other_index)

	ordered_candidates = [candidates[candidate_index] for candidate_index in kept_candidate_indices]
	ordered_candidates.sort(
		key=lambda candidate: (
			candidate.bounding_box[1],
			candidate.bounding_box[0],
			candidate.original_index,
		)
	)
	return [
		TextRegion(
			polygon_points=candidate.polygon_points,
			source_name=candidate.source_name,
			dedupe_priority=candidate.dedupe_priority,
		)
		for candidate in ordered_candidates
	]


def _candidate_preference(candidate: _RegionCandidate) -> tuple:
	return (
		candidate.dedupe_priority,
		candidate.point_count > 4,
		candidate.point_count,
		-candidate.area,
		-candidate.original_index,
	)


def _regions_overlap_enough(
	first_candidate: _RegionCandidate,
	second_candidate: _RegionCandidate,
	*,
	iou_threshold: float,
	overlap_threshold: float,
) -> bool:
	try:
		intersection_area = first_candidate.shapely_polygon.intersection(
			second_candidate.shapely_polygon
		).area
	except Exception:  # pragma: no cover - degenerate geometry
		return False

	if intersection_area <= 0.0:
		return False
	union_area = first_candidate.area + second_candidate.area - intersection_area
	polygon_iou = intersection_area / union_area if union_area > 0.0 else 0.0
	smaller_area = min(first_candidate.area, second_candidate.area)
	smaller_overlap = intersection_area / smaller_area if smaller_area > 0.0 else 0.0
	return polygon_iou >= iou_threshold or smaller_overlap >= overlap_threshold


def _bounding_boxes_overlap(first_box, second_box) -> bool:
	return not (
		first_box[2] < second_box[0]
		or second_box[2] < first_box[0]
		or first_box[3] < second_box[1]
		or second_box[3] < first_box[1]
	)
