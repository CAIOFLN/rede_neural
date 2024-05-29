import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Função para carregar e visualizar o dataset salvo
def visualize_saved_dataset(file_path, num_samples = 10):
    # Carregar o dataset salvo
    data_list = torch.load(file_path)
    
    # Verificar algumas amostras do dataset
    for i in range(min(num_samples, len(data_list))):
        img, boxes, labels, img_path = data_list[i]

        # Converter a imagem de tensor para PIL Image para visualização
        img = transforms.ToPILImage()(img)
        
        # Plotar a imagem
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        
        # Adicionar as bounding boxes
        for box in boxes:
            # COCO bounding boxes estão no formato [x, y, largura, altura]
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        plt.title(f'Sample {i+1}: {img_path}')
        plt.show()

# Caminho para o dataset salvo
file_path_train = 'coco_train_dataset.pt'
file_path_test = 'coco_test_dataset.pt'

# Visualizar algumas amostras do dataset de treino
visualize_saved_dataset(file_path_train)

# Visualizar algumas amostras do dataset de teste
visualize_saved_dataset(file_path_test)
