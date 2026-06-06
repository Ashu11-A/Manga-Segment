"""Modelo professor baseado no PaddleOCR para detecção de regiões de texto.

O PaddleOCR é usado **somente** como detector (``rec=False`` e ``cls=False``):
ele identifica onde há texto e devolve polígonos em pixels, mas nunca transcreve
o conteúdo. Assim, nenhum texto reconhecido vaza para os rótulos gerados —
exatamente o que o formato de máscara YOLO exige.

O detector roda no modo de polígono (``det_db_box_type='poly'``), produzindo
contornos com vários pontos que acompanham o formato do texto, em vez de simples
caixas de quatro cantos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
	"""Uma região de texto detectada, descrita por um polígono em pixels.

	``polygon_points`` é uma matriz ``numpy`` de forma ``(numero_de_pontos, 2)``
	com coordenadas absolutas ``(x, y)`` no espaço de pixels da imagem original.
	"""

	polygon_points: "object"  # numpy.ndarray de forma (N, 2); tipado tardiamente
	source_name: str = ""
	dedupe_priority: int = 0


class PaddleOcrTextTeacher:
	"""Encapsula o detector de texto do PaddleOCR acelerado por GPU.

	O modelo é carregado uma única vez na construção e reutilizado para todas as
	imagens, o que evita o custo alto de reinicialização a cada página.
	"""

	def __init__(
		self,
		*,
		language: str = "en",
		device: str = "auto",
		det_limit_side_len: int = 1536,
		det_db_unclip_ratio: float = 1.6,
		det_db_thresh: float = 0.3,
		det_db_box_thresh: float = 0.5,
		rotation_angles: list[int] | None = None,
		merge_iou_threshold: float = 0.5,
	) -> None:
		# A importação do PaddlePaddle/PaddleOCR é feita aqui (e não no topo do
		# módulo) para que o resto do framework não dependa de uma biblioteca
		# pesada que só é necessária durante a destilação.
		from paddleocr import PaddleOCR

		# Ângulos de aumento em tempo de inferência (rotation TTA): detectar a
		# página girada em 90°/270° "deita" o texto vertical do mangá, que o
		# detector DB (treinado em texto horizontal) então reconhece bem.
		self.rotation_angles = list(rotation_angles) if rotation_angles else [0]
		self.merge_iou_threshold = merge_iou_threshold

		self._paddle_ocr_class = PaddleOCR
		# ``det_limit_type='max'`` combinado com um lado grande preserva a
		# resolução de páginas de mangá, o que resulta em máscaras mais justas.
		# Limiares mais baixos (``det_db_thresh``/``det_db_box_thresh``) aumentam
		# o recall de texto fraco, rotacionado ou decorativo (efeitos sonoros).
		self._engine_kwargs = dict(
			lang=language,
			use_angle_cls=False,
			det_db_box_type="poly",
			det_limit_type="max",
			det_limit_side_len=det_limit_side_len,
			det_db_unclip_ratio=det_db_unclip_ratio,
			det_db_thresh=det_db_thresh,
			det_db_box_thresh=det_db_box_thresh,
			show_log=False,
		)

		requested_gpu = self._resolve_use_gpu(device)
		self._engine = self._build_engine(requested_gpu)
		self.use_gpu = requested_gpu

		# O wheel do PaddlePaddle-GPU do PyPI não traz o cuDNN; se a GPU foi
		# escolhida mas a inferência falha (ex.: cuDNN ausente), recarrega o
		# detector em CPU automaticamente, evitando erro a cada imagem.
		if requested_gpu and not self._gpu_inference_works():
			logger.warning(
				"Inferência na GPU indisponível (cuDNN/CUDA não carregou); "
				"recarregando o detector em CPU. Para usar a GPU, instale o "
				"cuDNN compatível ou rode com --device cpu para evitar este aviso."
			)
			self._engine = self._build_engine(use_gpu=False)
			self.use_gpu = False

		self.device_description = "GPU (CUDA)" if self.use_gpu else "CPU"

	def _build_engine(self, use_gpu: bool):
		"""Constrói uma instância do PaddleOCR para o dispositivo indicado."""
		return self._paddle_ocr_class(use_gpu=use_gpu, **self._engine_kwargs)

	def _gpu_inference_works(self) -> bool:
		"""Faz uma inferência de aquecimento para validar a pilha de GPU."""
		import numpy as np

		warmup_image = np.zeros((32, 32, 3), dtype=np.uint8)
		try:
			self._engine.ocr(warmup_image, det=True, rec=False, cls=False)
			return True
		except Exception as inference_error:  # pragma: no cover - depende da GPU
			logger.warning("Falha no aquecimento da GPU: %s", inference_error)
			return False

	@staticmethod
	def _resolve_use_gpu(requested_device: str) -> bool:
		"""Decide se a GPU será usada, respeitando a escolha do usuário.

		``auto`` usa GPU quando o PaddlePaddle foi compilado com CUDA e há ao
		menos um dispositivo visível; ``gpu`` força a GPU (com aviso e recuo
		para CPU se ela não existir); ``cpu`` desliga a GPU.
		"""
		normalized_device = (requested_device or "auto").lower()
		if normalized_device == "cpu":
			return False

		cuda_is_available = False
		try:
			import paddle

			cuda_is_available = (
				paddle.device.is_compiled_with_cuda()
				and paddle.device.cuda.device_count() > 0
			)
		except Exception:  # pragma: no cover - depende do ambiente
			cuda_is_available = False

		if normalized_device == "gpu" and not cuda_is_available:
			logger.warning(
				"GPU solicitada, mas CUDA não está disponível no PaddlePaddle; "
				"continuando em CPU."
			)
			return False
		if not cuda_is_available:
			return False

		if not PaddleOcrTextTeacher._cudnn_library_is_available():
			if normalized_device == "gpu":
				logger.warning(
					"GPU solicitada, mas a biblioteca cuDNN (libcudnn) não está "
					"carregável pelo sistema; continuando em CPU. Instale o cuDNN "
					"compatível com o wheel do PaddlePaddle-GPU ou rode com --device cpu."
				)
			return False

		return True

	@staticmethod
	def _cudnn_library_is_available() -> bool:
		"""Checks whether a cuDNN shared library can be loaded before GPU init."""
		import ctypes
		import ctypes.util

		candidate_library_names = [
			ctypes.util.find_library("cudnn"),
			"libcudnn.so",
			"libcudnn.so.9",
			"libcudnn.so.8",
		]
		for library_name in candidate_library_names:
			if not library_name:
				continue
			try:
				ctypes.CDLL(library_name)
				return True
			except OSError:
				continue
		return False

	def detect(self, image_bgr) -> list[TextRegion]:
		"""Detecta regiões de texto e devolve seus polígonos em pixels.

		Recebe uma imagem BGR (``numpy.ndarray``, como entregue pelo OpenCV) e
		retorna a lista de regiões. A recuperação de texto fica desligada, então
		o resultado contém apenas geometria.

		Quando há mais de um ângulo configurado (rotation TTA), a página é
		detectada em cada ângulo, os polígonos são levados de volta ao referencial
		original e as detecções duplicadas são fundidas por IoU.
		"""
		uses_tta = len(self.rotation_angles) > 1 or any(
			angle % 360 != 0 for angle in self.rotation_angles
		)
		if not uses_tta:
			collected_polygons = self._detect_single(image_bgr)
		else:
			collected_polygons = []
			for rotation_angle in self.rotation_angles:
				if rotation_angle % 360 == 0:
					# Sem rotação: detecta direto no referencial original.
					collected_polygons.extend(self._detect_single(image_bgr))
					continue
				rotated_image, rotation_matrix = self._rotate_image_expand(image_bgr, rotation_angle)
				for polygon_points in self._detect_single(rotated_image):
					collected_polygons.append(self._map_points_back(polygon_points, rotation_matrix))
			collected_polygons = self._merge_overlapping_polygons(collected_polygons)

		return [
			TextRegion(
				polygon_points=polygon_points,
				source_name="paddleocr",
				dedupe_priority=10,
			)
			for polygon_points in collected_polygons
			if polygon_points.shape[0] >= 3
		]

	def _detect_single(self, image_bgr) -> list:
		"""Roda uma única detecção e devolve polígonos ``(num_pontos, 2)`` válidos."""
		import numpy as np

		raw_detection_output = self._engine.ocr(image_bgr, det=True, rec=False, cls=False)
		valid_polygons: list = []
		for polygon_array in self._coerce_polygons(raw_detection_output):
			polygon_points = np.asarray(polygon_array, dtype=np.float32).reshape(-1, 2)
			if polygon_points.shape[0] >= 3:
				valid_polygons.append(polygon_points)
		return valid_polygons

	@staticmethod
	def _rotate_image_expand(image_bgr, rotation_angle: int):
		"""Gira a imagem em torno do centro, expandindo a tela para não cortar.

		Devolve a imagem girada e a matriz afim ``2x3`` usada, para que os
		polígonos detectados possam ser mapeados de volta ao referencial original.
		A borda é preenchida de branco (fundo típico de páginas de mangá).
		"""
		import cv2
		import numpy as np

		image_height, image_width = image_bgr.shape[:2]
		center_point = (image_width / 2.0, image_height / 2.0)
		rotation_matrix = cv2.getRotationMatrix2D(center_point, rotation_angle, 1.0)

		absolute_cos = abs(rotation_matrix[0, 0])
		absolute_sin = abs(rotation_matrix[0, 1])
		expanded_width = int(image_height * absolute_sin + image_width * absolute_cos)
		expanded_height = int(image_height * absolute_cos + image_width * absolute_sin)

		# Recoloca o centro para o meio da nova tela expandida.
		rotation_matrix[0, 2] += expanded_width / 2.0 - center_point[0]
		rotation_matrix[1, 2] += expanded_height / 2.0 - center_point[1]

		rotated_image = cv2.warpAffine(
			image_bgr,
			rotation_matrix,
			(expanded_width, expanded_height),
			flags=cv2.INTER_LINEAR,
			borderValue=(255, 255, 255),
		)
		return rotated_image, rotation_matrix.astype(np.float32)

	@staticmethod
	def _map_points_back(polygon_points, rotation_matrix):
		"""Mapeia pontos do referencial girado de volta ao original (afim inversa)."""
		import cv2
		import numpy as np

		inverse_matrix = cv2.invertAffineTransform(rotation_matrix)
		homogeneous_points = np.hstack(
			[polygon_points, np.ones((polygon_points.shape[0], 1), dtype=np.float32)]
		)
		return (homogeneous_points @ inverse_matrix.T).astype(np.float32)

	def _merge_overlapping_polygons(self, polygons: list) -> list:
		"""Funde detecções duplicadas (mesma região vista em ângulos diferentes).

		Faz um NMS por IoU de polígono usando ``shapely``: ordena por área
		decrescente, mantém o maior de cada grupo e suprime os que se sobrepõem
		acima de ``merge_iou_threshold``. Um pré-filtro por caixa envolvente evita
		o cálculo caro de IoU para pares que claramente não se tocam.
		"""
		if len(polygons) <= 1:
			return polygons
		try:
			from shapely.geometry import Polygon as ShapelyPolygon
		except Exception:  # pragma: no cover - shapely faz parte do paddleocr
			return polygons

		shapely_polygons = []
		bounding_boxes = []
		for polygon_points in polygons:
			shapely_polygon = ShapelyPolygon(polygon_points)
			if not shapely_polygon.is_valid:
				shapely_polygon = shapely_polygon.buffer(0)
			shapely_polygons.append(shapely_polygon)
			bounding_boxes.append(
				(
					float(polygon_points[:, 0].min()),
					float(polygon_points[:, 1].min()),
					float(polygon_points[:, 0].max()),
					float(polygon_points[:, 1].max()),
				)
			)

		order_by_area = sorted(
			range(len(polygons)),
			key=lambda index: shapely_polygons[index].area,
			reverse=True,
		)
		suppressed_indices: set[int] = set()
		kept_polygons: list = []
		for primary_index in order_by_area:
			if primary_index in suppressed_indices:
				continue
			kept_polygons.append(polygons[primary_index])
			for other_index in order_by_area:
				if other_index == primary_index or other_index in suppressed_indices:
					continue
				if not _bounding_boxes_overlap(
					bounding_boxes[primary_index], bounding_boxes[other_index]
				):
					continue
				if (
					_polygon_iou(shapely_polygons[primary_index], shapely_polygons[other_index])
					> self.merge_iou_threshold
				):
					suppressed_indices.add(other_index)
		return kept_polygons

	@staticmethod
	def _coerce_polygons(raw_detection_output) -> list:
		"""Normaliza as várias formas de retorno do PaddleOCR em uma lista plana.

		Dependendo da versão, ``ocr(..., rec=False)`` pode devolver
		``[[caixa, caixa, ...]]``, ``[caixa, ...]`` ou um ``ndarray`` de forma
		``(num_caixas, num_pontos, 2)``. Esta função desce recursivamente pela
		estrutura e coleta cada polígono folha de forma ``(num_pontos, 2)``.
		"""
		import numpy as np

		collected_polygons: list = []

		def visit(node) -> None:
			if node is None:
				return
			array_view = None
			try:
				array_view = np.asarray(node, dtype=np.float32)
			except (ValueError, TypeError):
				array_view = None

			# Folha: matriz (num_pontos, 2) — é um polígono completo.
			if array_view is not None and array_view.ndim == 2 and array_view.shape[1] == 2:
				collected_polygons.append(array_view)
				return
			# Bloco de polígonos: (num_caixas, num_pontos, 2).
			if array_view is not None and array_view.ndim == 3 and array_view.shape[2] == 2:
				for single_box in array_view:
					collected_polygons.append(single_box)
				return
			# Caso contrário, continua descendo na estrutura aninhada.
			if isinstance(node, (list, tuple)):
				for child_node in node:
					visit(child_node)

		visit(raw_detection_output)
		return collected_polygons


def _bounding_boxes_overlap(first_box, second_box) -> bool:
	"""Pré-filtro barato: as caixas envolventes axis-aligned se tocam?"""
	return not (
		first_box[2] < second_box[0]
		or second_box[2] < first_box[0]
		or first_box[3] < second_box[1]
		or second_box[3] < first_box[1]
	)


def _polygon_iou(first_polygon, second_polygon) -> float:
	"""IoU entre dois polígonos ``shapely`` (0.0 quando inválido/sem união)."""
	try:
		intersection_area = first_polygon.intersection(second_polygon).area
		union_area = first_polygon.area + second_polygon.area - intersection_area
	except Exception:  # pragma: no cover - geometria degenerada
		return 0.0
	if union_area <= 0.0:
		return 0.0
	return intersection_area / union_area
