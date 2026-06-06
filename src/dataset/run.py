#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "paddleocr>=2.7,<3.0",
#   "paddlepaddle-gpu>=2.6",
#   "opencv-python>=4.8",
#   "numpy<2",
#   "rich>=13",
#   "setuptools",
#   "pyyaml>=6",
#   "shapely>=2.0",
#   "pyclipper>=1.3",
# ]
# ///
"""Executável isolado da destilação de texto → dataset YOLO de segmentação.

Por que um script à parte? O PaddlePaddle fixa ``protobuf<=3.20.2``, o que
conflita com o ``tensorflow`` do projeto (exige ``protobuf>=3.20.3``); por isso
as duas pilhas não convivem no mesmo ambiente. Este arquivo usa metadados de
script embutidos (PEP 723), de modo que ``uv run`` cria um ambiente isolado só
com os professores de texto — sem tocar no ambiente principal do framework.

Uso:

    uv run src/dataset/run.py --show
    uv run src/dataset/run.py --teacher paddleocr --show
	uv run src/dataset/run.py --teacher ctd --dest manga-segment-text-ctd.v20i.yolo
	uv run src/dataset/run.py --teacher paddleocr,ctd --dest manga-segment-text-ensemble.v20i.yolo
    uv run src/dataset/run.py --source manga-segment.v20i.yolo26 \\
        --dest manga-segment-text.v20i.yolo --lang japan

Observação sobre GPU: o PaddleOCR também precisa que ``libcudnn`` esteja
instalado e carregável. Se CUDA/cuDNN não estiverem disponíveis, o professor
PaddleOCR recua para CPU; use ``--device cpu`` para evitar o aviso ou instale o
cuDNN/wheel adequado pelo índice oficial do Paddle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Coloca ``src/`` no caminho de importação para que ``dataset`` e ``core`` sejam
# importáveis mesmo executando este arquivo diretamente de dentro de src/dataset.
SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
	sys.path.insert(0, str(SOURCE_ROOT))

from dataset.distill import add_distill_arguments, run_distill  # noqa: E402


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		prog="distill",
		description="Destila um detector de texto professor em um dataset YOLO-seg.",
	)
	add_distill_arguments(parser)
	return run_distill(parser.parse_args(argv))


if __name__ == "__main__":
	raise SystemExit(main())
