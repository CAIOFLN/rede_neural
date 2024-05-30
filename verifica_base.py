import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Carregar dados do JSON
with open('simple_coco_ann_train.json', 'r') as f:
    data = json.load(f)
# Função para desenhar bounding boxes nas imagens

def draw_bboxes(image_info, annotations):
    img_path = 'base_de_dados/train/' + image_info['file_name']
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for ann in annotations:
        if ann['image_id'] == image_info['img_id']:
            bbox = ann['bbox']
            x, y, w, h = bbox
            draw.rectangle([x, y, x+w, y+h], outline="red", width=2)

    return img

# Mostrar imagens com bounding boxes
def show_images_with_bboxes(data, num_images=5):
    images = data['images']
    annotations = data['annotations']
    selected_images = images[:num_images]

    plt.figure(figsize=(15, 15))

    for i, image_info in enumerate(selected_images):
        img_with_bboxes = draw_bboxes(image_info, annotations)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img_with_bboxes)
        plt.axis('off')
        plt.title(f"Image ID: {image_info['img_id']}")

    plt.show()

# Executar a função para mostrar imagens com bounding boxes
show_images_with_bboxes(data, num_images=10)
