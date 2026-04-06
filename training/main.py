import argparse
from yolo.model import YoloModel 

parser = argparse.ArgumentParser(description="Gerenciador de Treinamento e Inferência YOLO")

parser.add_argument("--test", action="store_true", help="Testar o modelo gerando as inferências")
parser.add_argument("--best", action="store_true", help="Executar tuning para encontrar os melhores hiperparâmetros")
parser.add_argument("--convert", action="store_true", help="Converter um modelo treinado para TFJS/ONNX")
parser.add_argument("--fine-tune", action="store_true", help="Realizar fine-tuning em um modelo existente")

parser.add_argument("--model", type=int, help="ID do modelo base (model_id)")
parser.add_argument("--size", type=int, default=1280, help="Tamanho da imagem de entrada")
parser.add_argument("--epochs", type=int, help="Quantidade de épocas para treinamento ou fine-tuning")
parser.add_argument("--batch", type=int, default=1, help="Tamanho do lote (batch size)")
parser.add_argument("--patience", type=int, default=25, help="Paciência para o early stopping")

args = parser.parse_args()

yolo_manager = YoloModel(model_id=args.model, size=args.size)

if args.test:
  if not args.model:
    raise ValueError("⚠️ É necessário informar o argumento --model para testar.")
  yolo_manager.test()

elif args.convert:
  if not args.model:
    raise ValueError("⚠️ É necessário informar o argumento --model para converter.")
  converted_path = yolo_manager.convert()
  print(f"✅ Modelo convertido com sucesso. Os arquivos estão em: {converted_path}")

elif args.best:
  yolo_manager.tuning()

elif args.fine_tune:
  if not args.model:
    raise ValueError("⚠️ É necessário informar o argumento --model para realizar fine-tuning.")
  
  epochs_to_run = args.epochs if args.epochs is not None else 100
  yolo_manager.fine_tuning(epochs=epochs_to_run)

else:
  yolo_manager.train()