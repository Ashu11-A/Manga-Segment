import json

# Função para contar anotações por categoria
def contar_anotacoes_por_categoria(caminho_json, categorias_desejadas):
    with open(caminho_json, 'r') as f:
        data = json.load(f)
    
    # Criar um mapeamento de nome para ID das categorias desejadas
    categoria_ids = {cat['id'] for cat in data.get('categories', []) if cat['name'] in categorias_desejadas}
    
    # Contar as anotações que pertencem às categorias desejadas
    total_anotacoes = sum(1 for ann in data.get('annotations', []) if ann['category_id'] in categoria_ids)
    
    return total_anotacoes

# Caminhos para os arquivos de anotações
train_annotations = "/home/ashu/Documents/GitHub/Manga-Convert/model/train/_annotations.coco.json"
val_annotations = "/home/ashu/Documents/GitHub/Manga-Convert/model/valid/_annotations.coco.json"  # Se existir para COCO
test_annotations = "/home/ashu/Documents/GitHub/Manga-Convert/model/test/_annotations.coco.json"

# Categorias desejadas
categorias_desejadas = ['comic', 'speech-balloon']

# Contando as anotações por categoria
train_count = contar_anotacoes_por_categoria(train_annotations, categorias_desejadas)
val_count = contar_anotacoes_por_categoria(val_annotations, categorias_desejadas) if val_annotations else 0
test_count = contar_anotacoes_por_categoria(test_annotations, categorias_desejadas) if test_annotations else 0

# Exibindo os resultados
print(f"Total de anotações para as categorias {categorias_desejadas}: {train_count + val_count + test_count}")
