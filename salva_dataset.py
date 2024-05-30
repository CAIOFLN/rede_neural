import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import json

# Caminhos para o dataset COCO
ann_file_train = '_annotations.coco_train.json'
img_folder_train = 'base_de_dados/train'

# Definir transformações para as imagens
class ToTensor(object):
    def __call__(self, image):
        # Converte a imagem para tensor
        image = transforms.ToTensor()(image)
        return image

# Definir o Dataset personalizado para PyTorch
class COCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        # Inicializa o dataset com o caminho das imagens e do arquivo de anotações
        self.root = root
        with open(annFile, 'r') as f:
            self.data = json.load(f)
        self.transform = transform  # Armazena as transformações a serem aplicadas

    def __len__(self):
        # Retorna o número total de imagens no dataset
        return len(self.data['images'])

    def __getitem__(self, idx):
        # Obtém a imagem e suas anotações pelo índice

        # Carrega as informações da imagem
        img_info = self.data['images'][idx]

        # Cria o caminho completo da imagem
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')  # Abre a imagem e converte para RGB

        # Inicializa listas vazias para as anotações
        boxes = []
        labels = []

        # Verifica se há anotações para a imagem
        if 'annotations' in self.data:
            for ann in self.data['annotations']:
                if ann['image_id'] == img_info['id']:
                    bbox = ann['bbox']
                    boxes.append(bbox)
                    # Suponha que todos os objetos são da mesma classe para simplificar
                    labels.append(1)  # Atribui rótulo 1 para todas as classes

        # Se não houver anotações, adicionar caixas e rótulos vazios
        if not boxes:
            boxes.append([0, 0, 0, 0])  # Adicionar uma caixa vazia
            labels.append(0)  # Atribuir rótulo 0 para uma classe vazia

        # Converter para formato padrão PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Aplica a transformação, se houver
        if self.transform:
            img = self.transform(img)

        return {'img': img, 'boxes': boxes, 'labels': labels, 'img_path': img_path}

# Define a transformação de conversão para tensor
transform = ToTensor()

# Cria o dataset com a transformação definida
dataset_train = COCODataset(root=img_folder_train, annFile=ann_file_train, transform=transform)

# Função para salvar o dataset em disco em formato .pt
def save_dataset(dataset, file_path):
    torch.save(dataset, file_path)

# Salvar dataset de treino
save_dataset(dataset_train, 'coco_train_dataset.pt')
