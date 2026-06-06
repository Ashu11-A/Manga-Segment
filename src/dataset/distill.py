"""Orquestração da destilação de texto → dataset YOLO de segmentação.

Fluxo geral:

1. Descobre a estrutura de splits do dataset de origem lendo o sistema de
   arquivos (qualquer subpasta com um diretório ``images/`` vira um split), de
   modo que ``train``/``valid``/``test`` sejam tratados conforme existirem.
2. Carrega o professor de texto uma única vez.
3. Para cada imagem: detecta as regiões de texto, normaliza os polígonos,
   escreve o rótulo ``.txt`` (apenas coordenadas), copia a imagem para o dataset
   de destino e emite uma prévia visual em tempo real.
4. Escreve ``data.yaml`` e ``README.roboflow.txt`` no destino.

Somente a biblioteca padrão é importada no topo do módulo; ``numpy``, ``cv2`` e
os módulos irmãos são importados dentro das funções para que a construção da CLI
não puxe dependências pesadas.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("dataset.distill")

SUPPORTED_TEACHERS = {"paddleocr", "ctd"}

# Extensões de imagem aceitas ao varrer os diretórios de cada split.
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Mapeia nomes de pasta de split para as chaves esperadas pelo YOLO no data.yaml.
SPLIT_NAME_TO_YAML_KEY = {
	"train": "train",
	"valid": "val",
	"val": "val",
	"test": "test",
}


@dataclass
class DistillationConfig:
	"""Parâmetros de uma execução de destilação."""

	source_dataset: Path
	destination_dataset: Path
	teacher_name: str = "ensemble"
	ensemble_teacher_names: list[str] = field(default_factory=lambda: ["paddleocr", "ctd"])
	ensemble_iou_threshold: float = 0.5
	ensemble_overlap_threshold: float = 0.8
	class_id: int = 0
	class_name: str = "text"
	language: str = "en"
	device: str = "auto"
	det_limit_side_len: int = 1536
	det_db_unclip_ratio: float = 1.6
	det_db_thresh: float = 0.3
	det_db_box_thresh: float = 0.5
	# Rotation TTA: ângulos em que a página é detectada para recuperar texto
	# vertical/rotacionado; [0] desliga o aumento.
	rotation_tta_angles: list[int] = field(default_factory=lambda: [0, 90, 180, 270])
	merge_iou_threshold: float = 0.5
	ctd_model_path: Path | None = None
	ctd_model_url: str | None = None
	ctd_input_size: int = 1024
	ctd_polygon_mode: str = "mask"
	ctd_mask_thresh: float = 0.3
	ctd_box_thresh: float = 0.4
	ctd_unclip_ratio: float = 1.5
	ctd_contour_epsilon_ratio: float = 0.002
	ctd_max_candidates: int = 1000
	min_polygon_area: float = 10.0
	min_polygon_short_side: float = 3.0
	min_polygon_long_side: float = 8.0
	tiny_polygon_short_side: float = 6.0
	max_tiny_polygon_aspect_ratio: float = 40.0
	coordinate_precision: int = 6
	overwrite_existing: bool = False
	save_previews: bool = True
	preview_directory: Path | None = None
	show_window: bool = False
	# Quando falso (padrão), preserva os rótulos originais (comic/speech-balloon)
	# e apenas anexa a nova classe de texto. Quando verdadeiro, gera um dataset
	# de classe única só com o texto.
	text_only: bool = False


@dataclass
class SplitLayout:
	"""Estrutura descoberta de um split do dataset de origem."""

	split_name: str
	images_directory: Path
	image_paths: list[Path]


def discover_split_layouts(source_dataset_directory: Path) -> list[SplitLayout]:
	"""Descobre os splits varrendo o dataset de origem.

	Cada subpasta que contenha um diretório ``images/`` com ao menos uma imagem
	é considerada um split. Isso adapta o script automaticamente ao layout real
	(por exemplo, ``train`` + ``valid``, com ``test`` aparecendo se existir).
	"""
	discovered_splits: list[SplitLayout] = []
	if not source_dataset_directory.is_dir():
		return discovered_splits

	for child_directory in sorted(source_dataset_directory.iterdir()):
		if not child_directory.is_dir():
			continue
		images_directory = child_directory / "images"
		if not images_directory.is_dir():
			continue
		image_paths = sorted(
			candidate_path
			for candidate_path in images_directory.iterdir()
			if candidate_path.is_file()
			and candidate_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
		)
		if image_paths:
			discovered_splits.append(
				SplitLayout(
					split_name=child_directory.name,
					images_directory=images_directory,
					image_paths=image_paths,
				)
			)
	return discovered_splits


def read_source_class_names(source_dataset_directory: Path) -> list[str]:
	"""Lê a lista de classes do ``data.yaml`` do dataset de origem.

	Devolve, por exemplo, ``['comic', 'speech-balloon']`` para que a nova classe
	de texto seja anexada (e não substitua) as classes existentes. Retorna lista
	vazia quando não há ``data.yaml`` ou ``names`` legível.
	"""
	data_yaml_path = source_dataset_directory / "data.yaml"
	if not data_yaml_path.is_file():
		return []
	try:
		import yaml

		parsed_content = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
	except Exception:  # pragma: no cover - data.yaml malformado
		return []
	if not isinstance(parsed_content, dict):
		return []
	class_names = parsed_content.get("names")
	# ``names`` pode vir como lista ou como mapa {0: 'a', 1: 'b'}.
	if isinstance(class_names, dict):
		class_names = [class_names[key] for key in sorted(class_names)]
	if not isinstance(class_names, list):
		return []
	return [str(name) for name in class_names]


def build_label_line(
	polygon_points,
	image_width: int,
	image_height: int,
	class_id: int,
	coordinate_precision: int,
) -> str:
	"""Converte um polígono em pixels em uma linha de rótulo YOLO normalizada.

	A linha contém apenas ``class_id`` seguido das coordenadas normalizadas
	(``x1 y1 ... xn yn``); nenhum texto reconhecido é incluído. Cada coordenada é
	limitada ao intervalo ``[0, 1]`` para tolerar pequenos extravasamentos.
	"""
	normalized_coordinates: list[str] = []
	for point_x, point_y in polygon_points:
		normalized_x = min(max(float(point_x) / image_width, 0.0), 1.0)
		normalized_y = min(max(float(point_y) / image_height, 0.0), 1.0)
		normalized_coordinates.append(f"{normalized_x:.{coordinate_precision}f}")
		normalized_coordinates.append(f"{normalized_y:.{coordinate_precision}f}")
	return f"{class_id} " + " ".join(normalized_coordinates)


def write_dataset_descriptor(
	destination_dataset: Path,
	discovered_split_names: list[str],
	class_names: list[str],
) -> Path:
	"""Escreve o ``data.yaml`` do dataset de destino com a lista de classes.

	Os caminhos de split seguem o mesmo estilo do export original do Roboflow
	(``../<split>/images``), o que mantém compatibilidade na reimportação.
	"""
	descriptor_lines: list[str] = []
	for split_name in discovered_split_names:
		yaml_key = SPLIT_NAME_TO_YAML_KEY.get(split_name, split_name)
		descriptor_lines.append(f"{yaml_key}: ../{split_name}/images")

	names_literal = "[" + ", ".join(f"'{name}'" for name in class_names) + "]"
	descriptor_lines.append("")
	descriptor_lines.append(f"nc: {len(class_names)}")
	descriptor_lines.append(f"names: {names_literal}")
	descriptor_lines.append("")

	data_yaml_path = destination_dataset / "data.yaml"
	data_yaml_path.write_text("\n".join(descriptor_lines), encoding="utf-8")

	# Pequeno aviso de proveniência para acompanhar o dataset destilado.
	readme_path = destination_dataset / "README.roboflow.txt"
	readme_path.write_text(
		"Dataset gerado por destilação de modelo(s) professor(es) de texto.\n"
		f"Formato: YOLO instance segmentation. Classes: {names_literal}.\n"
		"Pronto para importação no Roboflow.\n",
		encoding="utf-8",
	)
	return data_yaml_path


def _display_is_available() -> bool:
	"""Indica se há um servidor gráfico para abrir uma janela ao vivo."""
	import os

	if os.name == "nt":
		return True
	return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _build_teacher(config: DistillationConfig):
	"""Build the selected text detector teacher once per distillation run."""
	normalized_teacher_name = config.teacher_name.lower()
	if normalized_teacher_name == "ensemble":
		from .ensemble_teacher import EnsembleTextTeacher

		teacher_models = [
			(teacher_name, _build_single_teacher(config, teacher_name))
			for teacher_name in config.ensemble_teacher_names
		]
		return EnsembleTextTeacher(
			teacher_models,
			iou_threshold=config.ensemble_iou_threshold,
			overlap_threshold=config.ensemble_overlap_threshold,
		)
	return _build_single_teacher(config, normalized_teacher_name)


def _build_single_teacher(config: DistillationConfig, teacher_name: str):
	"""Build one concrete teacher used directly or as part of an ensemble."""
	normalized_teacher_name = teacher_name.lower()
	if normalized_teacher_name == "paddleocr":
		from .paddleocr_teacher import PaddleOcrTextTeacher

		return PaddleOcrTextTeacher(
			language=config.language,
			device=config.device,
			det_limit_side_len=config.det_limit_side_len,
			det_db_unclip_ratio=config.det_db_unclip_ratio,
			det_db_thresh=config.det_db_thresh,
			det_db_box_thresh=config.det_db_box_thresh,
			rotation_angles=config.rotation_tta_angles,
			merge_iou_threshold=config.merge_iou_threshold,
		)
	if normalized_teacher_name == "ctd":
		from .ctd_teacher import ComicTextDetectorTeacher

		return ComicTextDetectorTeacher(
			model_path=config.ctd_model_path,
			model_url=config.ctd_model_url,
			input_size=config.ctd_input_size,
			polygon_mode=config.ctd_polygon_mode,
			mask_thresh=config.ctd_mask_thresh,
			box_thresh=config.ctd_box_thresh,
			unclip_ratio=config.ctd_unclip_ratio,
			contour_epsilon_ratio=config.ctd_contour_epsilon_ratio,
			max_candidates=config.ctd_max_candidates,
		)
	raise ValueError(f"professor desconhecido: {teacher_name}")


def _describe_teacher_config(config: DistillationConfig) -> str:
	if config.teacher_name.lower() == "ensemble":
		return (
			f" | modelos: {config.ensemble_teacher_names}"
			f" | CTD: {config.ctd_polygon_mode}"
			f" | dedupe IoU: {config.ensemble_iou_threshold}"
			f" | overlap: {config.ensemble_overlap_threshold}"
		)
	if config.teacher_name.lower() == "paddleocr":
		return f" | rotation TTA: {config.rotation_tta_angles}"
	if config.teacher_name.lower() == "ctd":
		return f" | polygon mode: {config.ctd_polygon_mode}"
	return ""


def generate_distilled_dataset(config: DistillationConfig) -> dict:
	"""Executa a destilação completa e devolve um resumo com os contadores."""
	import cv2

	from .dashboard import LiveDashboard, ProcessingMonitor
	from .overlay import draw_polygons_overlay

	discovered_splits = discover_split_layouts(config.source_dataset)
	if not discovered_splits:
		raise FileNotFoundError(
			"Nenhum split com pasta 'images/' foi encontrado em "
			f"{config.source_dataset}."
		)

	total_image_count = sum(len(split.image_paths) for split in discovered_splits)
	logger.info(
		"Origem: %s | Destino: %s | Splits: %s | Imagens: %d",
		config.source_dataset,
		config.destination_dataset,
		", ".join(split.split_name for split in discovered_splits),
		total_image_count,
	)

	# Decide o conjunto de classes do destino. Por padrão, preserva as classes
	# originais (comic/speech-balloon) e anexa a nova classe de texto ao final.
	existing_class_names = read_source_class_names(config.source_dataset)
	if config.text_only or not existing_class_names:
		destination_class_names = [config.class_name]
		text_class_id = config.class_id
		merge_source_labels = False
	elif config.class_name in existing_class_names:
		destination_class_names = list(existing_class_names)
		text_class_id = existing_class_names.index(config.class_name)
		merge_source_labels = True
	else:
		destination_class_names = list(existing_class_names) + [config.class_name]
		text_class_id = len(existing_class_names)
		merge_source_labels = True
	logger.info(
		"Classes do destino: %s | classe de texto = id %d | mesclar rótulos = %s",
		destination_class_names,
		text_class_id,
		merge_source_labels,
	)

	# Carrega o professor uma única vez (custo alto de inicialização).
	teacher_model = _build_teacher(config)
	teacher_details = _describe_teacher_config(config)
	logger.info(
		"Professor '%s' pronto em %s%s",
		config.teacher_name,
		teacher_model.device_description,
		teacher_details,
	)

	monitor = ProcessingMonitor(images_total=total_image_count)
	processed_split_names: list[str] = []
	# A janela ao vivo só é tentada quando solicitada e há display disponível.
	window_enabled = config.show_window and _display_is_available()

	with LiveDashboard(total_image_count) as dashboard:
		for split in discovered_splits:
			processed_split_names.append(split.split_name)

			destination_images_directory = config.destination_dataset / split.split_name / "images"
			destination_labels_directory = config.destination_dataset / split.split_name / "labels"
			destination_images_directory.mkdir(parents=True, exist_ok=True)
			destination_labels_directory.mkdir(parents=True, exist_ok=True)

			# Diretório dos rótulos originais deste split (irmão de images/).
			source_labels_directory = split.images_directory.parent / "labels"

			preview_directory: Path | None = None
			if config.save_previews:
				preview_root = config.preview_directory or (config.destination_dataset / "previews")
				preview_directory = preview_root / split.split_name
				preview_directory.mkdir(parents=True, exist_ok=True)

			for image_path in split.image_paths:
				label_path = destination_labels_directory / f"{image_path.stem}.txt"

				# Modo retomável: pula imagens já rotuladas, salvo --overwrite.
				if not config.overwrite_existing and label_path.exists():
					monitor.record_skipped()
					dashboard.update(monitor, split.split_name, image_path.name)
					continue

				try:
					polygon_count = _process_single_image(
						config=config,
						teacher_model=teacher_model,
						draw_overlay=draw_polygons_overlay,
						cv2_module=cv2,
						image_path=image_path,
						label_path=label_path,
						destination_images_directory=destination_images_directory,
						source_labels_directory=source_labels_directory,
						text_class_id=text_class_id,
						merge_source_labels=merge_source_labels,
						preview_directory=preview_directory,
						window_enabled=window_enabled,
					)
					monitor.record_success(polygon_count)
					logger.info(
						"[%s] %s -> %d polígono(s).",
						split.split_name,
						image_path.name,
						polygon_count,
					)
				except _WindowUnavailable:
					# O backend do OpenCV não suporta janelas; desliga e segue.
					window_enabled = False
					logger.warning("Janela ao vivo indisponível neste OpenCV; usando apenas prévias em disco.")
				except Exception:
					monitor.record_failure()
					logger.exception("Falha ao processar %s", image_path)

				dashboard.update(monitor, split.split_name, image_path.name)

	if window_enabled:
		cv2.destroyAllWindows()

	data_yaml_path = write_dataset_descriptor(
		config.destination_dataset,
		processed_split_names,
		destination_class_names,
	)

	summary = {
		"images_total": total_image_count,
		"images_processed": monitor.images_processed,
		"images_skipped": monitor.images_skipped,
		"images_failed": monitor.images_failed,
		"polygons_total": monitor.polygons_total,
		"empty_images": monitor.empty_images,
		"data_yaml": str(data_yaml_path),
		"elapsed_seconds": round(monitor.elapsed_seconds, 1),
	}
	logger.warning(
		"Concluído: %d processadas, %d puladas, %d falhas, %d polígonos em %.0fs.",
		summary["images_processed"],
		summary["images_skipped"],
		summary["images_failed"],
		summary["polygons_total"],
		summary["elapsed_seconds"],
	)
	return summary


class _WindowUnavailable(Exception):
	"""Sinaliza que ``cv2.imshow`` não é suportado pelo build atual do OpenCV."""


def _polygon_passes_size_filter(config: DistillationConfig, cv2_module, polygon_points) -> bool:
	"""Reject tiny/sliver detections that are disproportionate to real text."""
	polygon_area = abs(cv2_module.contourArea(polygon_points))
	if polygon_area < config.min_polygon_area:
		return False

	axis_aligned_width = float(polygon_points[:, 0].max() - polygon_points[:, 0].min())
	axis_aligned_height = float(polygon_points[:, 1].max() - polygon_points[:, 1].min())
	if max(axis_aligned_width, axis_aligned_height) < config.min_polygon_long_side:
		return False

	_, rotated_size, _ = cv2_module.minAreaRect(polygon_points.astype("float32"))
	rotated_width, rotated_height = abs(float(rotated_size[0])), abs(float(rotated_size[1]))
	short_side = min(rotated_width, rotated_height)
	long_side = max(rotated_width, rotated_height)
	if long_side < config.min_polygon_long_side:
		return False
	if short_side < config.min_polygon_short_side:
		return False

	# Very thin polygons are usually panel borders, screentone noise, or OCR
	# fragments. Keep this ratio high so legitimate vertical manga text survives.
	if (
		config.max_tiny_polygon_aspect_ratio > 0.0
		and short_side < config.tiny_polygon_short_side
		and long_side / max(short_side, 1e-6) > config.max_tiny_polygon_aspect_ratio
	):
		return False

	return True


def _process_single_image(
	*,
	config: DistillationConfig,
	teacher_model,
	draw_overlay,
	cv2_module,
	image_path: Path,
	label_path: Path,
	destination_images_directory: Path,
	source_labels_directory: Path,
	text_class_id: int,
	merge_source_labels: bool,
	preview_directory: Path | None,
	window_enabled: bool,
) -> int:
	"""Processa uma imagem; devolve a quantidade de polígonos de texto escritos."""
	image_bgr = cv2_module.imread(str(image_path))
	if image_bgr is None:
		raise ValueError(f"não foi possível ler a imagem: {image_path}")

	image_height, image_width = image_bgr.shape[:2]

	label_lines: list[str] = []
	# Preserva os rótulos originais (comic/speech-balloon) quando mesclando.
	if merge_source_labels:
		source_label_path = source_labels_directory / f"{image_path.stem}.txt"
		if source_label_path.is_file():
			for existing_line in source_label_path.read_text(encoding="utf-8").splitlines():
				if existing_line.strip():
					label_lines.append(existing_line.strip())

	# Detecta o texto e anexa os polígonos com a id da nova classe de texto.
	detected_regions = teacher_model.detect(image_bgr)
	kept_polygons: list = []
	for region in detected_regions:
		polygon_points = region.polygon_points
		if polygon_points.shape[0] < 3:
			continue
		if not _polygon_passes_size_filter(config, cv2_module, polygon_points):
			continue
		kept_polygons.append(polygon_points)
		label_lines.append(
			build_label_line(
				polygon_points,
				image_width,
				image_height,
				text_class_id,
				config.coordinate_precision,
			)
		)

	# Escreve o rótulo (arquivo vazio só quando não há nem texto nem original).
	label_text = "\n".join(label_lines)
	if label_lines:
		label_text += "\n"
	label_path.write_text(label_text, encoding="utf-8")

	# Copia a imagem original para o dataset de destino, preservando o nome.
	shutil.copy2(image_path, destination_images_directory / image_path.name)

	# Visualização em tempo real: prévia em disco e/ou janela ao vivo.
	if preview_directory is not None or window_enabled:
		annotated_image = draw_overlay(image_bgr, kept_polygons)
		if preview_directory is not None:
			cv2_module.imwrite(str(preview_directory / f"{image_path.stem}.jpg"), annotated_image)
		if window_enabled:
			try:
				cv2_module.imshow("Destilacao - mascaras de texto", annotated_image)
				cv2_module.waitKey(1)
			except cv2_module.error as opencv_error:
				raise _WindowUnavailable() from opencv_error

	return len(kept_polygons)


# --------------------------------------------------------------------------- #
# Integração com a linha de comando (subcomando ``distill`` em main.py).
# --------------------------------------------------------------------------- #


def add_distill_arguments(subparser: argparse.ArgumentParser) -> None:
	"""Adiciona os argumentos do subcomando ``distill`` ao parser fornecido."""
	subparser.add_argument(
		"--teacher",
		default="ensemble",
		help=(
			"Modelo professor: ensemble (padrão), paddleocr, ctd, "
			"ou lista separada por vírgula (ex.: paddleocr,ctd)."
		),
	)
	subparser.add_argument(
		"--teachers",
		default="paddleocr,ctd",
		help="Modelos usados quando --teacher ensemble, separados por vírgula.",
	)
	subparser.add_argument(
		"--ensemble-iou",
		type=float,
		default=0.5,
		help="IoU mínimo para considerar duas detecções do ensemble como duplicadas.",
	)
	subparser.add_argument(
		"--ensemble-overlap",
		type=float,
		default=0.8,
		help="Interseção/menor-área mínima para remover fragmentos duplicados.",
	)
	subparser.add_argument(
		"--source",
		default="manga-segment.v20i.yolo26",
		help="Dataset de origem (nome em dataset/ ou caminho absoluto).",
	)
	subparser.add_argument(
		"--dest",
		default="manga-segment-text.v20i.yolo",
		help="Dataset de destino (nome em dataset/ ou caminho absoluto).",
	)
	subparser.add_argument(
		"--class-name",
		default="text",
		help="Nome da nova classe anexada às classes existentes do dataset.",
	)
	subparser.add_argument(
		"--text-only",
		action="store_true",
		help="Gera um dataset de classe única (só texto), sem mesclar os rótulos originais.",
	)
	subparser.add_argument(
		"--class-id",
		type=int,
		default=0,
		help="Id da classe quando --text-only (ignorado ao mesclar; a id é derivada).",
	)
	subparser.add_argument("--lang", default="en", help="Idioma do PaddleOCR (ex.: en, japan, ml).")
	subparser.add_argument(
		"--device",
		default="auto",
		choices=["auto", "gpu", "cpu"],
		help="Dispositivo de inferência do professor.",
	)
	subparser.add_argument(
		"--det-limit-side-len",
		type=int,
		default=1536,
		help="Maior lado usado pelo detector (resolução das máscaras).",
	)
	subparser.add_argument(
		"--det-db-unclip-ratio",
		type=float,
		default=1.6,
		help="Fator de expansão do contorno detectado (DB unclip ratio).",
	)
	subparser.add_argument(
		"--det-db-thresh",
		type=float,
		default=0.3,
		help="Limiar de binarização do DB; menor detecta texto mais fraco/rotacionado.",
	)
	subparser.add_argument(
		"--det-db-box-thresh",
		type=float,
		default=0.5,
		help="Limiar de confiança da caixa; menor aumenta o recall (mais ruído).",
	)
	subparser.add_argument(
		"--tta-angles",
		default="0,90,180,270",
		help="Ângulos do rotation TTA, separados por vírgula (recupera texto vertical).",
	)
	subparser.add_argument(
		"--no-rotation-tta",
		action="store_true",
		help="Desliga o rotation TTA (detecta apenas em 0°).",
	)
	subparser.add_argument(
		"--merge-iou",
		type=float,
		default=0.5,
		help="Limiar de IoU para fundir detecções duplicadas entre ângulos.",
	)
	subparser.add_argument(
		"--ctd-model-path",
		default=None,
		help="Caminho do modelo CTD ONNX; se ausente, baixa/cacheia em models/ctd/.",
	)
	subparser.add_argument(
		"--ctd-model-url",
		default=None,
		help="URL alternativa para baixar o modelo CTD ONNX quando o arquivo não existe.",
	)
	subparser.add_argument(
		"--ctd-input-size",
		type=int,
		default=1024,
		help="Tamanho quadrado de entrada do CTD ONNX.",
	)
	subparser.add_argument(
		"--ctd-polygon-mode",
		choices=["mask", "line_box"],
		default="mask",
		help="Como converter CTD em polígonos: mask segue o texto; line_box usa caixas rotacionadas.",
	)
	subparser.add_argument(
		"--ctd-mask-thresh",
		type=float,
		default=0.3,
		help="Limiar do mapa de máscara/DB do CTD antes de extrair contornos.",
	)
	subparser.add_argument(
		"--ctd-box-thresh",
		type=float,
		default=0.4,
		help="Limiar de confiança dos polígonos CTD em --ctd-polygon-mode line_box.",
	)
	subparser.add_argument(
		"--ctd-unclip-ratio",
		type=float,
		default=1.5,
		help="Expansão dos polígonos DB em --ctd-polygon-mode line_box.",
	)
	subparser.add_argument(
		"--ctd-contour-epsilon-ratio",
		type=float,
		default=0.002,
		help="Simplificação dos contornos de máscara CTD; menor preserva mais pontos.",
	)
	subparser.add_argument(
		"--ctd-max-candidates",
		type=int,
		default=1000,
		help="Máximo de contornos CTD avaliados por imagem.",
	)
	subparser.add_argument(
		"--min-polygon-area",
		type=float,
		default=10.0,
		help="Área mínima (em pixels) para manter um polígono.",
	)
	subparser.add_argument(
		"--min-polygon-short-side",
		type=float,
		default=3.0,
		help="Menor lado mínimo (px) do retângulo rotacionado do texto.",
	)
	subparser.add_argument(
		"--min-polygon-long-side",
		type=float,
		default=8.0,
		help="Maior lado mínimo (px) do retângulo rotacionado do texto.",
	)
	subparser.add_argument(
		"--tiny-polygon-short-side",
		type=float,
		default=6.0,
		help="Abaixo deste menor lado (px), aplica o filtro de proporção extrema.",
	)
	subparser.add_argument(
		"--max-tiny-polygon-aspect-ratio",
		type=float,
		default=40.0,
		help="Proporção máxima para polígonos muito finos; 0 desliga este filtro.",
	)
	subparser.add_argument(
		"--precision",
		type=int,
		default=6,
		help="Casas decimais das coordenadas normalizadas.",
	)
	subparser.add_argument(
		"--overwrite",
		action="store_true",
		help="Reprocessa imagens cujo rótulo já existe (desliga o modo retomável).",
	)
	subparser.add_argument(
		"--no-preview",
		action="store_true",
		help="Não salva as prévias visuais em disco.",
	)
	subparser.add_argument(
		"--preview-dir",
		default=None,
		help="Diretório das prévias (padrão: <dest>/previews).",
	)
	subparser.add_argument(
		"--show",
		action="store_true",
		help="Abre uma janela ao vivo com cada página anotada (requer display).",
	)
	subparser.add_argument(
		"--log-file",
		default=None,
		help="Arquivo de log (padrão: logs/text_masks_<timestamp>.log).",
	)
	subparser.add_argument("--log-level", default="INFO", help="Nível de log do arquivo.")


def _resolve_dataset_path(path_value: str) -> Path:
	"""Resolve um nome de dataset dentro de ``dataset/`` ou um caminho absoluto."""
	candidate_path = Path(path_value).expanduser()
	if candidate_path.is_absolute():
		return candidate_path
	from core import paths

	return Path(paths.dataset_path(path_value))


def _default_log_file() -> Path:
	"""Caminho padrão do arquivo de log, sob ``logs/`` na raiz do repositório."""
	from core import paths

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return Path(paths.REPO_ROOT) / "logs" / f"text_masks_{timestamp}.log"


def _parse_csv_values(raw_values: str) -> list[str]:
	"""Parse a comma-separated CLI value, preserving first occurrence order."""
	parsed_values: list[str] = []
	seen_values: set[str] = set()
	for raw_value in raw_values.split(","):
		value = raw_value.strip().lower()
		if value and value not in seen_values:
			parsed_values.append(value)
			seen_values.add(value)
	return parsed_values


def _normalize_teacher_selection(args: argparse.Namespace) -> tuple[str, list[str]]:
	"""Resolve --teacher/--teachers into one mode plus concrete teacher names."""
	requested_teacher = str(args.teacher or "ensemble").strip().lower()
	if "," in requested_teacher:
		teacher_names = _parse_csv_values(requested_teacher)
		teacher_mode = "ensemble"
	elif requested_teacher == "ensemble":
		teacher_names = _parse_csv_values(args.teachers)
		teacher_mode = "ensemble"
	else:
		teacher_names = [requested_teacher]
		teacher_mode = requested_teacher

	invalid_teacher_names = [
		teacher_name for teacher_name in teacher_names if teacher_name not in SUPPORTED_TEACHERS
	]
	if teacher_mode not in SUPPORTED_TEACHERS | {"ensemble"} or invalid_teacher_names:
		raise ValueError(
			"professor inválido; use ensemble, paddleocr, ctd ou uma lista com esses nomes"
		)
	if teacher_mode == "ensemble" and not teacher_names:
		raise ValueError("--teacher ensemble requer ao menos um nome em --teachers")
	return teacher_mode, teacher_names


def run_distill(args: argparse.Namespace) -> int:
	"""Ponto de entrada do subcomando ``distill`` chamado por main.py."""
	from .dashboard import configure_logging

	source_dataset = _resolve_dataset_path(args.source)
	destination_dataset = _resolve_dataset_path(args.dest)

	log_file = Path(args.log_file) if args.log_file else _default_log_file()
	configure_logging(log_file, args.log_level)
	logger.info("Logs detalhados em: %s", log_file)

	# Sem TTA -> só 0°; caso contrário, converte "0,90,180,270" em [0, 90, 180, 270].
	if args.no_rotation_tta:
		rotation_tta_angles = [0]
	else:
		rotation_tta_angles = [
			int(angle_text.strip())
			for angle_text in args.tta_angles.split(",")
			if angle_text.strip()
		] or [0]
	teacher_name, ensemble_teacher_names = _normalize_teacher_selection(args)

	configuration = DistillationConfig(
		source_dataset=source_dataset,
		destination_dataset=destination_dataset,
		teacher_name=teacher_name,
		ensemble_teacher_names=ensemble_teacher_names,
		ensemble_iou_threshold=args.ensemble_iou,
		ensemble_overlap_threshold=args.ensemble_overlap,
		class_id=args.class_id,
		class_name=args.class_name,
		language=args.lang,
		device=args.device,
		det_limit_side_len=args.det_limit_side_len,
		det_db_unclip_ratio=args.det_db_unclip_ratio,
		det_db_thresh=args.det_db_thresh,
		det_db_box_thresh=args.det_db_box_thresh,
		rotation_tta_angles=rotation_tta_angles,
		merge_iou_threshold=args.merge_iou,
		ctd_model_path=Path(args.ctd_model_path).expanduser() if args.ctd_model_path else None,
		ctd_model_url=args.ctd_model_url,
		ctd_input_size=args.ctd_input_size,
		ctd_polygon_mode=args.ctd_polygon_mode,
		ctd_mask_thresh=args.ctd_mask_thresh,
		ctd_box_thresh=args.ctd_box_thresh,
		ctd_unclip_ratio=args.ctd_unclip_ratio,
		ctd_contour_epsilon_ratio=args.ctd_contour_epsilon_ratio,
		ctd_max_candidates=args.ctd_max_candidates,
		min_polygon_area=args.min_polygon_area,
		min_polygon_short_side=args.min_polygon_short_side,
		min_polygon_long_side=args.min_polygon_long_side,
		tiny_polygon_short_side=args.tiny_polygon_short_side,
		max_tiny_polygon_aspect_ratio=args.max_tiny_polygon_aspect_ratio,
		coordinate_precision=args.precision,
		overwrite_existing=args.overwrite,
		save_previews=not args.no_preview,
		preview_directory=Path(args.preview_dir) if args.preview_dir else None,
		show_window=args.show,
		text_only=args.text_only,
	)

	summary = generate_distilled_dataset(configuration)
	print(
		f"✅ {summary['images_processed']} imagem(ns) processada(s), "
		f"{summary['polygons_total']} polígono(s). data.yaml: {summary['data_yaml']}"
	)
	return 0
