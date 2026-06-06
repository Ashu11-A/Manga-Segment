"""Geração de vídeos por classe a partir de um dataset YOLO de segmentação.

Para cada classe do dataset, este módulo:

1. percorre os rótulos (``labels/*.txt``) de TODOS os splits (train, valid e
   test, quando existir), lê os polígonos de segmentação normalizados e, por
   padrão, usa TODAS as instâncias da classe (use ``--count`` para limitar);
2. recorta um quadrado centrado no objeto (com ``--margin`` de respiro) e o
   coloca SEMPRE no centro de uma tela fixa, preenchendo com fundo quando o
   recorte ultrapassa a borda da página — todos os quadros ficam registrados
   (um sobre o outro), deixando o corte seco entre instâncias bem mais suave;
3. monta um vídeo por classe com corte seco (um quadro por instância), usando o
   ``ffmpeg``.

Os quadros são escritos em disco à medida que são gerados (streaming), então o
processamento de todas as imagens não depende de manter tudo na memória. Ao
final, imprime quais classes têm aparência média mais parecida.

A camada de orquestração (``run_class_videos``) e o construtor de argumentos
(``add_class_videos_arguments``) são reexportados para que ``main.py`` exponha o
comando ``class-videos``. O ``opencv``/``numpy`` só são importados dentro das
funções, mantendo a construção da linha de comando leve. O módulo também roda de
forma autônoma::

    uv run src/dataset/class_videos.py                       # todas as imagens
    uv run src/dataset/class_videos.py --count 100 --fps 8   # amostra de 100
    uv run src/dataset/class_videos.py --classes text --margin 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Permite rodar este arquivo diretamente (uv run src/dataset/class_videos.py),
# colocando ``src/`` no caminho de importação para alcançar ``core``.
SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
	sys.path.insert(0, str(SOURCE_ROOT))

DEFAULT_DATASET = "manga-segment_v2-v7-yolo26"
DEFAULT_SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
BACKGROUND_COLORS = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}
# Descritor por quadro usado para comparar CLASSES (só no relatório de classes
# parecidas). Combina três blocos que continuam significativos ao se tirar a
# média entre quadros: histograma de orientações de borda (forma) + histograma de
# intensidades (tom) + estatísticas de borda. O recorte é reduzido a FEATURE_SIZE².
FEATURE_SIZE = 64
ORIENT_BINS = 12
INTENSITY_BINS = 16


@dataclass(frozen=True)
class Instance:
	"""Uma instância de uma classe: a imagem de origem e o bounding box (px)."""

	image_path: Path
	box: tuple[int, int, int, int]  # (x0, y0, x1, y1) em pixels


def add_class_videos_arguments(parser: argparse.ArgumentParser) -> None:
	"""Registra os argumentos do comando ``class-videos`` no parser dado."""
	parser.add_argument(
		"--dataset",
		default=None,
		help=(
			"Pasta do dataset YOLO (com data.yaml e os splits). Aceita caminho "
			f"absoluto ou relativo a ./dataset (padrão: {DEFAULT_DATASET})."
		),
	)
	parser.add_argument(
		"--output",
		default=None,
		help="Pasta de saída dos quadros e vídeos (padrão: previews/class_videos).",
	)
	parser.add_argument(
		"--count",
		type=int,
		default=None,
		help="Limite de instâncias por classe (padrão: todas as imagens do dataset).",
	)
	parser.add_argument(
		"--canvas",
		type=int,
		default=640,
		help="Lado, em pixels, da tela quadrada de cada quadro (padrão: 640).",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=5.0,
		help="Quadros por segundo do vídeo gerado (padrão: 5).",
	)
	parser.add_argument(
		"--margin",
		type=float,
		default=0.1,
		help=(
			"Respiro ao redor do objeto, como fração do seu maior lado: o recorte "
			"é um quadrado de lado max(w,h)*(1+2*margin) centrado no objeto "
			"(padrão: 0.1). Maior = mais contexto da página em volta."
		),
	)
	parser.add_argument(
		"--bg",
		choices=sorted(BACKGROUND_COLORS),
		default="black",
		help="Cor de fundo (usada apenas com --mask ou em recortes nas bordas).",
	)
	parser.add_argument(
		"--mask",
		action="store_true",
		help="Preenche tudo fora do polígono com a cor de fundo (isola o objeto).",
	)
	parser.add_argument(
		"--classes",
		nargs="*",
		default=None,
		help="Subconjunto de classes a processar (padrão: todas do data.yaml).",
	)
	parser.add_argument(
		"--splits",
		nargs="*",
		default=list(DEFAULT_SPLITS),
		help="Splits a varrer em busca de instâncias (padrão: train valid test).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Semente para embaralhar a ordem dos quadros (reprodutibilidade).",
	)
	parser.add_argument(
		"--keep-frames",
		action="store_true",
		help="Mantém os quadros .jpg intermediários em disco (padrão: apaga).",
	)


def _resolve_dataset_dir(dataset: str | None) -> Path:
	"""Resolve o caminho do dataset (absoluto, ou relativo a ./dataset)."""
	from core import paths

	name = dataset or DEFAULT_DATASET
	candidate = Path(name).expanduser()
	if candidate.is_absolute() and candidate.exists():
		return candidate
	# Tenta como caminho relativo ao diretório de trabalho e depois a ./dataset.
	if candidate.exists():
		return candidate.resolve()
	return Path(paths.dataset_path(name))


def _load_class_names(dataset_dir: Path) -> list[str]:
	"""Lê os nomes das classes do data.yaml (com fallback para PyYAML ausente)."""
	data_yaml = dataset_dir / "data.yaml"
	if not data_yaml.exists():
		raise FileNotFoundError(f"data.yaml não encontrado em {dataset_dir}")
	try:
		import yaml

		data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
		names = data.get("names")
		if isinstance(names, dict):  # formato {0: nome, ...}
			return [names[key] for key in sorted(names)]
		if isinstance(names, list):
			return list(names)
	except Exception:
		pass
	# Fallback simples: procura a linha "names: [...]".
	for line in data_yaml.read_text(encoding="utf-8").splitlines():
		stripped = line.strip()
		if stripped.startswith("names:") and "[" in stripped:
			inside = stripped[stripped.index("[") + 1 : stripped.rindex("]")]
			return [item.strip().strip("'\"") for item in inside.split(",") if item.strip()]
	raise ValueError(f"Não foi possível ler 'names' de {data_yaml}")


def _find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
	"""Encontra a imagem correspondente a um arquivo de rótulo .txt."""
	stem = label_path.stem
	for extension in IMAGE_EXTENSIONS:
		candidate = images_dir / f"{stem}{extension}"
		if candidate.exists():
			return candidate
	return None


def _polygon_points(tokens: list[str]) -> list[tuple[float, float]]:
	"""Converte os tokens de uma linha de rótulo em pares (x, y) normalizados.

	Suporta tanto o formato de segmentação (vários pares de pontos) quanto o de
	detecção (``cx cy w h``), devolvendo neste caso os quatro cantos da caixa.
	"""
	values = [float(value) for value in tokens]
	if len(values) == 4:  # caixa de detecção: cx, cy, w, h
		cx, cy, width, height = values
		x0, y0 = cx - width / 2, cy - height / 2
		x1, y1 = cx + width / 2, cy + height / 2
		return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
	pairs = list(zip(values[0::2], values[1::2]))
	return pairs


def _collect_instances(
	dataset_dir: Path,
	splits: list[str],
	class_count: int,
) -> dict[int, list[Instance]]:
	"""Varre os splits e agrupa todas as instâncias por id de classe."""
	import cv2

	# Cache de tamanho (largura, altura) por imagem, para não reler do disco.
	size_cache: dict[Path, tuple[int, int] | None] = {}

	def image_size(path: Path) -> tuple[int, int] | None:
		if path not in size_cache:
			image = cv2.imread(str(path))
			size_cache[path] = (image.shape[1], image.shape[0]) if image is not None else None
		return size_cache[path]

	per_class: dict[int, list[Instance]] = defaultdict(list)
	for split in splits:
		labels_dir = dataset_dir / split / "labels"
		images_dir = dataset_dir / split / "images"
		if not labels_dir.exists() or not images_dir.exists():
			continue
		for label_path in sorted(labels_dir.glob("*.txt")):
			lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
			if not lines:
				continue
			image_path = _find_image_for_label(label_path, images_dir)
			if image_path is None:
				continue
			size = image_size(image_path)
			if size is None:
				continue
			width, height = size
			for line in lines:
				tokens = line.split()
				class_id = int(float(tokens[0]))
				if class_id >= class_count:
					continue
				points = _polygon_points(tokens[1:])
				if len(points) < 2:
					continue
				xs = [point[0] * width for point in points]
				ys = [point[1] * height for point in points]
				x0, x1 = max(0, int(min(xs))), min(width, int(round(max(xs))))
				y0, y1 = max(0, int(min(ys))), min(height, int(round(max(ys))))
				if x1 - x0 < 2 or y1 - y0 < 2:  # descarta caixas degeneradas
					continue
				per_class[class_id].append(Instance(image_path, (x0, y0, x1, y1)))
	return per_class


def _centered_square(image, box, margin, canvas_size, bg_color):
	"""Coloca o objeto SEMPRE no centro de uma tela quadrada ``canvas_size``.

	Extrai um quadrado de lado ``max(w,h)*(1+2*margin)`` centrado no centro do
	objeto e o redimensiona para a tela. Diferentemente da versão anterior, o
	quadrado NÃO é deslocado para caber na página: quando ele ultrapassa a borda,
	a parte de fora é preenchida com ``bg_color``. Assim o centro do objeto cai
	exatamente no centro da tela em TODOS os quadros — os recortes ficam
	registrados (um sobre o outro) e o corte seco entre eles fica bem mais suave.
	"""
	import cv2
	import numpy as np

	height, width = image.shape[:2]
	x0, y0, x1, y1 = box
	center_x = (x0 + x1) / 2.0
	center_y = (y0 + y1) / 2.0
	side = int(round(max(2.0, max(x1 - x0, y1 - y0) * (1.0 + 2.0 * max(0.0, margin)))))
	sx0 = int(round(center_x - side / 2.0))
	sy0 = int(round(center_y - side / 2.0))

	square = np.full((side, side, 3), bg_color[::-1], dtype=np.uint8)  # fundo (BGR)
	# Interseção entre o quadrado (em coords da página) e a imagem real.
	ix0, iy0 = max(0, sx0), max(0, sy0)
	ix1, iy1 = min(width, sx0 + side), min(height, sy0 + side)
	if ix1 > ix0 and iy1 > iy0:
		square[iy0 - sy0 : iy1 - sy0, ix0 - sx0 : ix1 - sx0] = image[iy0:iy1, ix0:ix1]

	interpolation = cv2.INTER_AREA if side > canvas_size else cv2.INTER_CUBIC
	return cv2.resize(square, (canvas_size, canvas_size), interpolation=interpolation)


def _class_feature(canvas):
	"""Descritor de aparência por quadro, para comparar CLASSES.

	Combina três blocos que continuam significativos ao se tirar a média entre
	quadros: histograma de orientações de borda (forma — separa contornos curvos
	de balões, bordas retas de caixas e o traçado denso de texto), histograma de
	intensidades (tom) e estatísticas de borda (densidade e magnitude média).
	"""
	import cv2
	import numpy as np

	small = cv2.resize(canvas, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

	grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
	magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
	orientation = np.arctan2(grad_y, grad_x) % np.pi  # borda não tem sentido

	orient_bin = np.minimum(
		(orientation / np.pi * ORIENT_BINS).astype(np.int32), ORIENT_BINS - 1
	)
	orient_hist = np.bincount(
		orient_bin.ravel(), weights=magnitude.ravel(), minlength=ORIENT_BINS
	).astype(np.float32)
	orient_hist /= max(1e-6, float(orient_hist.sum()))

	intensity_bin = (gray.astype(np.int32) * INTENSITY_BINS) // 256
	intensity_hist = np.bincount(
		intensity_bin.ravel(), minlength=INTENSITY_BINS
	).astype(np.float32)
	intensity_hist /= max(1.0, float(intensity_hist.sum()))

	threshold = float(magnitude.mean() + magnitude.std())
	edge_density = float((magnitude > threshold).mean())
	mean_magnitude = float(magnitude.mean()) / 255.0
	scalars = np.array([edge_density, mean_magnitude], dtype=np.float32)

	return np.concatenate([orient_hist, intensity_hist, scalars])


def _render_class(
	instances: list[Instance],
	frames_dir: Path,
	*,
	canvas_size: int,
	margin: float,
	bg_color: tuple[int, int, int],
	apply_mask: bool,
	dataset_dir: Path,
	splits: list[str],
):
	"""Recorta/centraliza cada instância e escreve os quadros numerados em disco.

	Faz streaming: nenhuma lista de imagens é mantida em memória. As instâncias
	são agrupadas pela imagem de origem para que cada página seja lida (e, com
	--mask, mascarada) uma única vez. Devolve ``(quadros_salvos, descritor_medio)``.
	"""
	import cv2

	frames_dir.mkdir(parents=True, exist_ok=True)
	for existing in frames_dir.glob("frame_*.jpg"):
		existing.unlink()

	# Mapa stem -> caminho do rótulo, para mascarar pelo polígono quando pedido.
	label_lookup: dict[str, Path] = {}
	if apply_mask:
		for split in splits:
			labels_dir = dataset_dir / split / "labels"
			if labels_dir.exists():
				for label_path in labels_dir.glob("*.txt"):
					label_lookup.setdefault(label_path.stem, label_path)

	# Agrupa por imagem preservando a ordem (já embaralhada) de primeira aparição.
	by_image: dict[Path, list[Instance]] = defaultdict(list)
	for instance in instances:
		by_image[instance.image_path].append(instance)

	saved = 0
	descriptor_sum = None
	for image_path, items in by_image.items():
		image = cv2.imread(str(image_path))
		if image is None:
			continue
		if apply_mask:
			image = _mask_page(image, label_lookup.get(image_path.stem), bg_color)
		for instance in items:
			canvas = _centered_square(image, instance.box, margin, canvas_size, bg_color)
			saved += 1
			cv2.imwrite(str(frames_dir / f"frame_{saved:06d}.jpg"), canvas)
			feature = _class_feature(canvas)
			descriptor_sum = feature if descriptor_sum is None else descriptor_sum + feature

	mean_descriptor = descriptor_sum / saved if (descriptor_sum is not None and saved) else None
	return saved, mean_descriptor


def _mask_page(image, label_path, bg_color):
	"""Devolve a página inteira com tudo fora dos polígonos pintado de ``bg_color``.

	É computada uma vez por página (não por recorte); o recorte centralizado
	apenas amostra dessa página já mascarada.
	"""
	import cv2
	import numpy as np

	if label_path is None:
		return image
	height, width = image.shape[:2]
	mask = np.zeros((height, width), dtype=np.uint8)
	drew = False
	for line in label_path.read_text().splitlines():
		tokens = line.strip().split()
		if len(tokens) < 5:
			continue
		points = _polygon_points(tokens[1:])
		polygon = np.array(
			[[int(px * width), int(py * height)] for px, py in points], dtype=np.int32
		)
		cv2.fillPoly(mask, [polygon], 255)
		drew = True
	if not drew:
		return image
	background = np.full_like(image, bg_color[::-1])  # BGR
	return np.where(mask[:, :, None] > 0, image, background)


def _build_video(frames_dir: Path, output_path: Path, fps: float) -> bool:
	"""Monta o vídeo da classe a partir dos quadros usando ffmpeg."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if output_path.exists():
		output_path.unlink()
	command = [
		"ffmpeg",
		"-y",
		"-framerate",
		str(fps),
		"-i",
		str(frames_dir / "frame_%06d.jpg"),
		"-c:v",
		"libx264",
		"-pix_fmt",
		"yuv420p",
		# Garante dimensões pares (exigência do yuv420p), por segurança.
		"-vf",
		"pad=ceil(iw/2)*2:ceil(ih/2)*2",
		str(output_path),
	]
	result = subprocess.run(command, capture_output=True, text=True)
	if result.returncode != 0:
		print(result.stderr.strip().splitlines()[-1] if result.stderr else "", file=sys.stderr)
		return False
	return True


def _report_similar_classes(class_means: dict[str, object]) -> None:
	"""Aponta, para cada classe, a mais parecida, e ranqueia todos os pares.

	Cada classe é resumida pela MÉDIA do descritor de aparência dos seus quadros
	(forma + tom + borda). Os três blocos são normalizados em separado para
	contribuírem por igual e as classes são comparadas por similaridade de cosseno.
	"""
	import numpy as np

	names = list(class_means)
	if len(names) < 2:
		return
	matrix = np.stack([np.asarray(class_means[name], dtype=np.float32) for name in names])
	# Padroniza cada dimensão ENTRE as classes (z-score por coluna): realça as
	# características em que as classes de fato diferem e descarta as que são iguais
	# para todas — sem isso, recortes quase todos brancos ficam ~colineares e a
	# similaridade satura perto de 1. Depois compara por cosseno (≈ correlação).
	standardized = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-6)
	norms = np.linalg.norm(standardized, axis=1, keepdims=True)
	unit = standardized / np.maximum(norms, 1e-6)
	similarity = unit @ unit.T
	np.fill_diagonal(similarity, -np.inf)

	print("\n🔬 Classe mais parecida com cada uma:")
	for index, name in enumerate(names):
		best = int(similarity[index].argmax())
		print(f"   {name:16s} → {names[best]} ({similarity[index, best]:+.3f})")

	pairs = sorted(
		(
			(float(similarity[i, j]), names[i], names[j])
			for i in range(len(names))
			for j in range(i + 1, len(names))
		),
		reverse=True,
	)
	print("\n📊 Ranking de pares (cosseno):")
	for score, first, second in pairs:
		print(f"   {first} ↔ {second}: {score:+.3f}")


def run_class_videos(args: argparse.Namespace) -> int:
	"""Orquestra a extração de recortes e a geração de um vídeo por classe."""
	if shutil.which("ffmpeg") is None:
		print("❌ ffmpeg não encontrado no PATH. Instale o ffmpeg e tente de novo.", file=sys.stderr)
		return 1

	from core import paths

	dataset_dir = _resolve_dataset_dir(args.dataset)
	if not dataset_dir.exists():
		print(f"❌ Dataset não encontrado: {dataset_dir}", file=sys.stderr)
		return 1

	names = _load_class_names(dataset_dir)
	output_dir = Path(args.output) if args.output else Path(paths.REPO_ROOT) / "previews" / "class_videos"
	bg_color = BACKGROUND_COLORS[args.bg]

	# Seleciona quais classes processar (por nome) preservando a ordem do yaml.
	wanted = set(args.classes) if args.classes else None
	targets = [(i, n) for i, n in enumerate(names) if wanted is None or n in wanted]
	if wanted:
		unknown = wanted - {n for _, n in targets}
		if unknown:
			print(f"⚠️  Classes ignoradas (não existem no dataset): {', '.join(sorted(unknown))}")

	print(f"📂 Dataset: {dataset_dir}")
	print(f"🎬 Saída:   {output_dir}")
	print(f"🔎 Coletando instâncias em: {', '.join(args.splits)} ...")
	per_class = _collect_instances(dataset_dir, list(args.splits), len(names))

	rng = random.Random(args.seed)
	generated: list[Path] = []
	class_means: dict[str, object] = {}
	for class_id, class_name in targets:
		pool = per_class.get(class_id, [])
		if not pool:
			print(f"⚠️  [{class_name}] nenhuma instância encontrada — pulando.")
			continue
		sample = pool[:]
		rng.shuffle(sample)
		if args.count:
			sample = sample[: args.count]

		print(f"🖼️  [{class_name}] {len(sample)} instâncias — gerando quadros (corte seco) ...")
		frames_dir = output_dir / "frames" / class_name
		saved, mean_descriptor = _render_class(
			sample,
			frames_dir,
			canvas_size=args.canvas,
			margin=args.margin,
			bg_color=bg_color,
			apply_mask=args.mask,
			dataset_dir=dataset_dir,
			splits=list(args.splits),
		)
		if saved == 0:
			print(f"⚠️  [{class_name}] nenhum quadro válido — pulando vídeo.")
			continue
		if mean_descriptor is not None:
			class_means[class_name] = mean_descriptor

		video_path = output_dir / f"{class_name}.mp4"
		if _build_video(frames_dir, video_path, args.fps):
			generated.append(video_path)
			print(f"✅ [{class_name}] {saved} quadros → {video_path}")
		else:
			print(f"❌ [{class_name}] falha ao gerar o vídeo.", file=sys.stderr)

		if not args.keep_frames:
			shutil.rmtree(frames_dir, ignore_errors=True)

	_report_similar_classes(class_means)
	print(f"\n🏁 {len(generated)} vídeo(s) gerado(s) em {output_dir}")
	return 0 if generated else 1


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		prog="class-videos",
		description="Gera um vídeo por classe com recortes centralizados (corte seco).",
	)
	add_class_videos_arguments(parser)
	return run_class_videos(parser.parse_args(argv))


if __name__ == "__main__":
	raise SystemExit(main())
