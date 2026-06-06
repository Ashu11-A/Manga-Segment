"""Registro de logs e visualização em tempo real do processamento.

Três responsabilidades:

* ``configure_logging`` — direciona o log detalhado para um arquivo e mantém o
  console limpo (apenas avisos/erros), para não embaralhar o painel ao vivo.
* ``ProcessingMonitor`` — acumula os contadores de progresso (imagens, polígonos,
  páginas vazias, falhas) e calcula a vazão.
* ``LiveDashboard`` — painel ``rich`` atualizado a cada imagem, com recuo
  automático para uma barra ``tqdm`` quando o ``rich`` não está instalado.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_logging(log_file: Path, log_level: str = "INFO") -> Path:
	"""Configura o logger raiz com saída em arquivo e console enxuto.

	O arquivo recebe tudo no nível pedido (``INFO`` por padrão); o console fica
	limitado a ``WARNING`` para não competir com o painel ao vivo. Devolve o
	caminho do arquivo de log criado.
	"""
	log_file.parent.mkdir(parents=True, exist_ok=True)
	resolved_level = getattr(logging, str(log_level).upper(), logging.INFO)

	root_logger = logging.getLogger()
	root_logger.setLevel(resolved_level)
	# Remove handlers de execuções anteriores para evitar linhas duplicadas.
	for existing_handler in list(root_logger.handlers):
		root_logger.removeHandler(existing_handler)

	log_formatter = logging.Formatter(
		"%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
		datefmt="%H:%M:%S",
	)

	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setLevel(resolved_level)
	file_handler.setFormatter(log_formatter)
	root_logger.addHandler(file_handler)

	console_handler = logging.StreamHandler()
	# Console só mostra avisos/erros; o detalhe por imagem vai para o arquivo.
	console_handler.setLevel(logging.WARNING)
	console_handler.setFormatter(log_formatter)
	root_logger.addHandler(console_handler)

	return log_file


@dataclass
class ProcessingMonitor:
	"""Acumula os contadores de progresso da destilação."""

	images_total: int
	images_processed: int = 0
	images_skipped: int = 0
	images_failed: int = 0
	polygons_total: int = 0
	empty_images: int = 0
	start_timestamp: float = field(default_factory=time.monotonic)

	def record_success(self, polygon_count: int) -> None:
		"""Registra uma imagem processada com sucesso e seus polígonos."""
		self.images_processed += 1
		self.polygons_total += polygon_count
		if polygon_count == 0:
			self.empty_images += 1

	def record_skipped(self) -> None:
		"""Registra uma imagem pulada (rótulo já existente, sem sobrescrever)."""
		self.images_skipped += 1

	def record_failure(self) -> None:
		"""Registra uma imagem que falhou no processamento."""
		self.images_failed += 1

	@property
	def images_completed(self) -> int:
		"""Total de imagens já encerradas (processadas + puladas + falhas)."""
		return self.images_processed + self.images_skipped + self.images_failed

	@property
	def elapsed_seconds(self) -> float:
		"""Tempo decorrido desde o início, em segundos."""
		return max(time.monotonic() - self.start_timestamp, 1e-6)

	@property
	def images_per_second(self) -> float:
		"""Vazão média de imagens encerradas por segundo."""
		return self.images_completed / self.elapsed_seconds


class LiveDashboard:
	"""Painel de progresso em tempo real (``rich``) com recuo para ``tqdm``.

	É um gerenciador de contexto: entra/sai do ``Live`` do ``rich`` (ou fecha a
	barra do ``tqdm``) automaticamente. Quando nenhum dos dois está disponível,
	degrada silenciosamente para nenhuma saída visual.
	"""

	def __init__(self, total_images: int, *, prefer_rich: bool = True) -> None:
		self._total_images = max(int(total_images), 1)
		self._backend = "none"
		self._rich_live = None
		self._tqdm_bar = None
		self._current_split = ""
		self._current_filename = ""

		if prefer_rich and self._try_init_rich():
			self._backend = "rich"
		elif self._try_init_tqdm():
			self._backend = "tqdm"

	def _try_init_rich(self) -> bool:
		try:
			from rich.console import Console
			from rich.live import Live

			self._rich_console = Console()
			self._rich_live = Live(
				console=self._rich_console,
				refresh_per_second=8,
				transient=False,
			)
			return True
		except Exception:
			self._rich_live = None
			return False

	def _try_init_tqdm(self) -> bool:
		try:
			from tqdm import tqdm

			self._tqdm_bar = tqdm(total=self._total_images, unit="img", dynamic_ncols=True)
			return True
		except Exception:
			self._tqdm_bar = None
			return False

	def __enter__(self) -> "LiveDashboard":
		if self._backend == "rich":
			self._rich_live.__enter__()
		return self

	def __exit__(self, *exception_info) -> bool:
		if self._backend == "rich":
			self._rich_live.__exit__(*exception_info)
		elif self._backend == "tqdm":
			self._tqdm_bar.close()
		return False

	def update(self, monitor: ProcessingMonitor, split_name: str, current_filename: str) -> None:
		"""Atualiza o painel com o estado atual do monitor."""
		self._current_split = split_name
		self._current_filename = current_filename
		if self._backend == "rich":
			self._rich_live.update(self._render_rich_panel(monitor))
		elif self._backend == "tqdm":
			self._tqdm_bar.n = monitor.images_completed
			self._tqdm_bar.set_postfix(
				split=split_name,
				polygons=monitor.polygons_total,
				empty=monitor.empty_images,
				failed=monitor.images_failed,
				refresh=False,
			)
			self._tqdm_bar.refresh()

	def _render_rich_panel(self, monitor: ProcessingMonitor):
		from rich.panel import Panel
		from rich.table import Table

		statistics_table = Table.grid(padding=(0, 2))
		statistics_table.add_column(justify="right", style="bold cyan")
		statistics_table.add_column(justify="left")

		completion_percent = 100.0 * monitor.images_completed / self._total_images
		statistics_table.add_row("Split atual", str(self._current_split))
		statistics_table.add_row("Imagem atual", str(self._current_filename))
		statistics_table.add_row(
			"Progresso",
			f"{monitor.images_completed}/{self._total_images} ({completion_percent:.1f}%)",
		)
		statistics_table.add_row("Polígonos", str(monitor.polygons_total))
		statistics_table.add_row("Páginas vazias", str(monitor.empty_images))
		statistics_table.add_row("Puladas", str(monitor.images_skipped))
		statistics_table.add_row("Falhas", str(monitor.images_failed))
		statistics_table.add_row("Vazão", f"{monitor.images_per_second:.2f} img/s")
		statistics_table.add_row("Tempo", f"{monitor.elapsed_seconds:.0f}s")

		return Panel(
			statistics_table,
			title="Destilação de texto → YOLO-seg",
			border_style="green",
		)
