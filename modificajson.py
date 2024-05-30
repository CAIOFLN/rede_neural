import json

# Caminho para o arquivo JSON original
input_file = '/home/caio/Desktop/CNN/base_de_dados/_annotations.coco_train.json'

# Caminho para o arquivo JSON simplificado que será gerado
output_file = '_annotations.coco_train.json'

# Função para processar o arquivo JSON
def simplify_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    simplified_data = {
        'images': [],
        'annotations': []
    }

    # Dicionário para manter a anotação de maior área para cada imagem
    image_to_best_annotation = {}

    # Encontrando a anotação com maior área para cada imagem
    for ann in data['annotations']:
        image_id = ann['image_id']
        area = ann.get('area', 0)

        if image_id in image_to_best_annotation:
            current_best_area = image_to_best_annotation[image_id].get('area', 0)
            if area > current_best_area:
                image_to_best_annotation[image_id] = ann
        else:
            image_to_best_annotation[image_id] = ann

    # Adicionando as imagens e as anotações simplificadas
    for img in data['images']:
        image_id = img['id']

        if image_id in image_to_best_annotation:
            best_annotation = image_to_best_annotation[image_id]

            simplified_image = {
                'file_name': img['file_name'],
                'height': img['height'],
                'width': img['width']
            }
            simplified_data['images'].append(simplified_image)

            simplified_annotation = {
                'image_id': best_annotation['image_id'],
                'category_id': best_annotation['category_id'],
                'bbox': best_annotation['bbox'],
                'area': best_annotation.get('area', None)
            }
            simplified_data['annotations'].append(simplified_annotation)

    # Escrevendo o arquivo JSON simplificado
    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=2)

    print(f'Arquivo simplificado gerado em: {output_file}')

# Chamando a função para simplificar o arquivo JSON
simplify_json(input_file, output_file)
