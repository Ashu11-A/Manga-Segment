"""O plugin de algoritmo de segmentação YOLO."""

from __future__ import annotations

import os
import re
from typing import Any

import yaml

from core.algorithm import BaseAlgorithm
from core.device import resolve_device
from core.registry import register
from core.segmenter import BaseSegmenter

from algorithms.yolo import weights as weights_mod
from algorithms.yolo.models import DEFAULT_MODEL
from algorithms.yolo.segmenter import YoloSegmenter


@register
class YoloAlgorithm(BaseAlgorithm):
  """Backend de segmentação Ultralytics YOLO."""

  name = "yolo"
  display_name = "YOLO"

  def __init__(
    self,
    *,
    size: int | list[int] | None = None,
    model_id: int | None = None,
    model_dir: str | None = None,
    model_name: str = DEFAULT_MODEL,
    data_path: str | None = None,
    config_path: str | None = None,
  ) -> None:
    super().__init__(size=size, model_id=model_id, model_dir=model_dir)
    self.model_name = model_name
    # Resolved lazily (see _require_data_path): None means "auto-detect the
    # dataset under dataset/ when a command actually needs it".
    self.data_path = data_path
    self.config_path = config_path or os.path.join(
      os.path.dirname(__file__), "best_hyperparameters.yaml"
    )

  def _require_data_path(self) -> str:
    """Return the dataset data.yaml, auto-detecting it on first use."""
    from core.dataset import resolve_data_yaml

    self.data_path = resolve_data_yaml(self.data_path)
    return self.data_path

  def _resolve_size(self) -> int | list[int]:
    """Return the training image size, auto-detecting it from the dataset.

    When ``--size`` is left unset (``None``) the size is inferred from the
    dataset's own images the first time it is needed, then cached so subsequent
    calls (e.g. the ONNX export after training) reuse the same value.
    """
    if self.size is None:
      from core.dataset import detect_image_size

      self.size = detect_image_size(self._require_data_path())
      print(f"📐 Auto-detected image size from dataset: {self.size}")
    return self.size

  def resolve_model_ref(
    self,
    *,
    model_id: int | None = None,
    model_dir: str | None = None,
  ) -> str:
    return weights_mod.resolve_weights(model_id=model_id, model_dir=model_dir)

  def build_segmenter(
    self,
    weights: str,
    *,
    confidence: float = 0.5,
    augment: bool = False,
    **kwargs: Any,
  ) -> YoloSegmenter:
    # Inference does not depend on the dataset, so fall back to the YOLO default
    # rather than auto-detecting when --size was left unset.
    imgsz = self.size if self.size is not None else 1280
    return YoloSegmenter(weights, imgsz=imgsz, conf=confidence, augment=augment)

  def _update_model_id(self, model) -> None:
    trainer = model.trainer

    if trainer is None:
      print("⚠️ model.trainer não está definido. O treinamento falhou ou não foi concluído.")
      return

    save_dir = str(getattr(trainer, "save_dir", ""))

    if not save_dir:
      return

    folder_name = os.path.basename(save_dir)
    match_id = re.search(r'\d+$', folder_name)

    self.model_id = int(match_id.group()) if match_id else 1
    print(f'✅ model_id atualizado para: {self.model_id}')

  def train(self, **kwargs: Any) -> None:
    from ultralytics import YOLO

    device = resolve_device()
    size = self._resolve_size()

    if self.model_id is not None:
      model_path = weights_mod.resolve_weights(model_id=self.model_id)
      model = YOLO(model_path, task='segment').to(device)
    else:
      model = YOLO(self.model_name, task='segment').to(device)

    model.train(
      data=self._require_data_path(),
      cfg=self.config_path,
      patience=100,
      epochs=1000,
      batch=8,
      imgsz=size,
      cache=True,               # Carrega as imagens diretamente na RAM do sistema, eliminando gargalos de leitura de disco.
      optimizer="MuSGD",        # Define o algoritmo (Momentum SGD) usado para minimizar a perda e ajustar os pesos.
      rect=False,               # Desativa o preenchimento de imagens retangulares, liberando o processamento em grade quadrada do Mosaic.
      # copy_paste=0.3,           # Probabilidade (30%) de recortar as máscaras das classes e colá-las em novos fundos de outras imagens.
      # mixup=0.2,                # Probabilidade (20%) de sobrepor duas imagens transparentes, forçando aprendizado de padrões mistos.
      # mosaic=1.0,               # Ativação total (100%) da técnica que junta 4 imagens em uma única, aumentando o contexto de pequenos objetos.
      # multi_scale=0.25,         # Oscila a resolução de entrada dinamicamente em até +/- 25% para tornar a rede robusta a distâncias diferentes.
      # mask_ratio=2,             # Define a compressão da máscara de segmentação resultante (ex: 2 comprime para um quarto do tamanho).
      # dropout=0.1,              # Desliga 10% das conexões neurais aleatoriamente a cada época para impedir a rede de apenas memorizar os dados.
      val=True,                 # Habilita o cálculo do mAP nas imagens de validação para mensurar o desempenho real.
      plots=True,               # Instrui a biblioteca a renderizar as imagens de curva de perda, matrizes de confusão e detecções visuais.
      save=True,                # Permite que o modelo salve fisicamente os arquivos .pt do treinamento.
      save_period=50,           # Determina a criação de um backup de segurança do checkpoint atual a cada 50 épocas completadas.
    )

    self._update_model_id(model)
    model.export(format="onnx", imgsz=size, half=True)

  def tune(self, **kwargs: Any) -> None:
    from ultralytics import YOLO

    size = self._resolve_size()
    model = YOLO(self.model_name, task="segment")

    model.tune(
      data=self._require_data_path(),
      imgsz=size,
      epochs=100,
      iterations=1000,
      batch=-1,
      plots=False,
      save=False,
      val=False,
    )

  def fine_tune(self, epochs: int = 100, **kwargs: Any) -> None:
    from ultralytics import YOLO

    if self.model_id is None:
      raise ValueError("model_id não definido para fine-tuning.")

    device = resolve_device()
    size = self._resolve_size()
    model_path = weights_mod.resolve_weights(model_id=self.model_id)
    model = YOLO(model_path, task="segment").to(device)

    model.train(
      data=self._require_data_path(),
      cfg=self.config_path,
      epochs=epochs,
      batch=1,
      imgsz=size,
      optimizer="MuSGD",
      lr0=0.0001,
      cache=True,
    )

    self._update_model_id(model)
    model.export(format="onnx", imgsz=size, half=True)

  def convert(self, **kwargs: Any) -> str:
    from ultralytics import YOLO

    if self.model_id is None:
      raise ValueError("model_id não definido para conversão.")

    weights_path = weights_mod.resolve_weights(model_id=self.model_id)
    model_path = os.path.dirname(os.path.dirname(weights_path))

    with open(os.path.join(model_path, "args.yaml")) as file:
      model_size = yaml.safe_load(file)["imgsz"]

    if not isinstance(model_size, list) and not isinstance(model_size, int):
      raise ValueError(f"Não foi possível determinar o tamanho do modelo: {model_path}")

    model = YOLO(weights_path, task="segment")

    if not os.path.exists(os.path.join(model_path, "weights/best_web_model")):
      model.export(format="tfjs", imgsz=model_size, keras=True)

    return model_path

  def benchmark(self, **kwargs: Any) -> str:
    from algorithms.yolo import benchmark as benchmark_mod

    return benchmark_mod.run_benchmark(
      size=self._resolve_size(),
      data_path=self._require_data_path(),
      config_path=self.config_path,
      **kwargs,
    )

  def report(self, **kwargs: Any) -> str:
    from algorithms.yolo import benchmark as benchmark_mod

    return benchmark_mod.generate_report(data_path=self._require_data_path())