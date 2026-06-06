"""Geração de datasets por destilação de um modelo professor.

Este pacote distila o conhecimento de um ou mais modelos professores
(ensemble PaddleOCR + CTD por padrão) para dentro de um dataset no formato de
segmentação de instâncias do YOLO, pronto para importação no Roboflow.

Apenas a camada de orquestração (``distill``) é reexportada aqui. Os módulos
pesados (``paddleocr_teacher``, ``ctd_teacher``, ``ensemble_teacher``,
``overlay``, ``dashboard``) só são importados sob demanda, dentro das funções,
para que ``import dataset`` permaneça leve e não puxe ``opencv``/``paddle`` ao
construir a linha de comando.
"""

from __future__ import annotations

from .distill import (
	DistillationConfig,
	add_distill_arguments,
	generate_distilled_dataset,
	run_distill,
)
from .download import add_download_arguments, run_download

__all__ = [
	"DistillationConfig",
	"add_distill_arguments",
	"generate_distilled_dataset",
	"run_distill",
	"add_download_arguments",
	"run_download",
]
