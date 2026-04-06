from ultralytics import YOLO, checks
from yolo.utils import getModel
import os
import torch
import yaml
import re
import cv2 as cv
import numpy as np

checks()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class YoloModel:
  model_name: str
  model_id: int | None
  size: int | list[int]
  data_path: str
  config_path: str
  
  def __init__(
    self, 
    model_name: str = "yolo26s-seg.pt", 
    model_id: int | None = None, 
    size: int | list[int] = 1280,
    data_path: str = os.path.abspath("../dataset/yolo/data.yaml"),
    config_path: str = os.path.abspath("yolo/best_hyperparameters.yaml")
  ) -> None:
    self.model_id = model_id
    self.size = size
    self.model_name = model_name
    self.data_path = data_path
    self.config_path = config_path
  
  def _update_model_id(self, model: YOLO) -> None:
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

  @staticmethod
  def _apply_mask(image_data: np.ndarray, mask_data: np.ndarray) -> np.ndarray:
    mask_binary = np.where(mask_data > 0, 255, 0).astype(np.uint8)

    if image_data.shape[:2] != mask_binary.shape:
      mask_binary = cv.resize(mask_binary, (image_data.shape[1], image_data.shape[0]))

    bgr_image = cv.bitwise_and(image_data, image_data, mask=mask_binary)
    bgra_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2BGRA)
    bgra_image[:, :, 3] = mask_binary

    return bgra_image

  def train(self) -> None:
    if self.model_id is not None:
      model_path = getModel(model_num=self.model_id, find='weights/best.pt')
      model = YOLO(os.path.join(model_path, 'weights/best.pt'), task='segment').to(device)
    else:
      model = YOLO(self.model_name, task='segment').to(device)

    model.train(
      data=self.data_path,
      cfg=self.config_path,
      patience=100,
      epochs=1000,
      batch=5,
      imgsz=self.size,
      cache=True,
      optimizer="MuSGD",
      rect=True,          # Treinamento com as imagens em vertical
      multi_scale=0.25,   # Varia o tamanho da imagem de -25% a +25%
      mask_ratio=2,       # Qualidade da mascará, por default ela fica com um tamanho de 1/4 da imagem original
      dropout=0.1,       # Evitar overfitting no seu dataset pequeno
      
      # --- Parâmetros para visualizar o aprendizado ---
      val=True,           # Ativa a etapa de validação (necessário para gerar as imagens)
      plots=True,         # Força a geração dos gráficos e das imagens de segmentação
      save=True,          # Salva os pesos (weights) do modelo
      save_period=50,     # Opcional: Salva um arquivo '.pt' de backup a cada 50 épocas
      # -----------------------------------------------
    )

    self._update_model_id(model)
    model.export(format="onnx", imgsz=self.size, half=True)
    
  def tuning(self) -> None:
    model = YOLO(self.model_name, task="segment")

    model.tune(
      data=self.data_path,
      imgsz=self.size,
      epochs=100,
      iterations=1000,
      batch=-1,
      plots=False,
      save=False,
      val=False,
    )

  def fine_tuning(self, epochs: int = 100) -> None:
    if self.model_id is None:
      raise ValueError("model_id não definido para fine-tuning.")

    model_path = getModel(model_num=self.model_id, find="weights/best.pt")
    model = YOLO(os.path.join(model_path, "weights/best.pt"), task="segment").to(device)

    model.train(
      data=self.data_path,
      cfg=self.config_path,
      epochs=epochs,
      batch=1,
      imgsz=self.size,
      optimizer="MuSGD",
      lr0=0.0001, 
      cache=True,
    )

    self._update_model_id(model)
    model.export(format="onnx", imgsz=self.size, half=True)

  def convert(self) -> str:
    if self.model_id is None:
      raise ValueError("model_id não definido para conversão.")
    
    model_path = getModel(model_num=self.model_id, find="weights/best.pt")
    
    with open(os.path.join(model_path, "args.yaml")) as file:
      model_size = yaml.safe_load(file)["imgsz"]

    if not isinstance(model_size, list) and not isinstance(model_size, int):
      raise ValueError(f"Não foi possível determinar o tamanho do modelo: {model_path}")

    model = YOLO(os.path.join(model_path, "weights/best.pt"), task="segment")

    if not os.path.exists(os.path.join(model_path, "weights/best_web_model")):
      model.export(format="tfjs", imgsz=model_size, keras=True)

    return model_path

  def test(
    self, 
    images_dir: str = os.path.abspath("../images"), 
    output_dir: str = os.path.abspath("../output")
  ) -> None:
    if self.model_id is None:
      raise ValueError("model_id não definido para teste.")

    model_path = getModel(model_num=self.model_id, find="weights/best.pt")
    
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    model = YOLO(os.path.join(model_path, "weights/best.pt"), task="segment").to(device)

    for image_name in os.listdir(images_dir):
      image_path = os.path.join(images_dir, image_name)

      if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

      image_data = cv.imread(image_path)
      if image_data is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        continue

      results = model.predict(
        source=image_data,
        save=False,
        imgsz=self.size,
        conf=0.5,
        task="segment",
        half=True,
        augment=True,
        agnostic_nms=True,
        retina_masks=True,
      )

      for result in results:
        if result.masks is None or result.boxes is None:
          continue
          
        masks_tensor = result.masks.data
        detected_classes_tensor = result.boxes.cls
        class_names = result.names

        # Validação de tipo explícita para resolver o alerta do Pylance
        masks_data = masks_tensor.cpu().numpy() if isinstance(masks_tensor, torch.Tensor) else masks_tensor
        detected_classes = detected_classes_tensor.cpu().numpy() if isinstance(detected_classes_tensor, torch.Tensor) else detected_classes_tensor

        for mask_index, mask_array in enumerate(masks_data):
          class_name = class_names[int(detected_classes[mask_index])]
          processed_mask = (mask_array * 255).astype(np.uint8)
          
          file_base_name = os.path.splitext(image_name)[0]
          mask_file_path = os.path.join(output_dir, f"{file_base_name}_{class_name}_mask.png")
          cv.imwrite(mask_file_path, processed_mask)

        combined_mask = np.any(masks_data, axis=0).astype(np.uint8) * 255
        masked_image = self._apply_mask(image_data, combined_mask)
        
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_segmented.png")
        cv.imwrite(output_file_path, masked_image)

      print(f"Processamento concluído: {image_name}")
