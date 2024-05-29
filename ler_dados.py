import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pycocotools.coco import COCO

# Caminhos para o dataset COCO
ann_file_teste = 'base_de_dados/_annotations.coco_teste.json'
ann_file_train = 'base_de_dados/_annotations.coco_train.json'

# Caminhos para a pasta que contém as imagens
img_folder_teste = 'base_de_dados/test'
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
        self.coco = COCO(annFile)  # Carrega as anotações COCO
        self.ids = list(sorted(self.coco.imgs.keys()))  # Obtém e ordena os IDs das imagens
        self.transform = transform  # Armazena as transformações a serem aplicadas

    def __len__(self):
        # Retorna o número total de imagens no dataset
        return len(self.ids)

    def __getitem__(self, idx):
        # Obtém a imagem e suas anotações pelo índice

        # Obtém o ID da imagem
        img_id = self.ids[idx]

        # Obtém as IDs das anotações conforme o ID da imagem
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # Carrega as anotações
        anns = self.coco.loadAnns(ann_ids)

        # Carrega as informações da imagem
        img_info = self.coco.loadImgs(img_id)[0]

        # Cria o caminho completo da imagem
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')  # Abre a imagem e converte para RGB

        # Obter as bounding boxes e rótulos
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(ann['bbox'])  # Adiciona a bounding box
            # Suponha que todos os objetos são da mesma classe para simplificar
            labels.append(1)  # Atribui rótulo 1 para todas as classes

        # Converter para formato padrão PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Aplica a transformação, se houver
        if self.transform:
            img = self.transform(img)

        return img, boxes, labels, img_path  # Retorna a imagem, as caixas delimitadoras, os rótulos e o caminho da imagem

# Define a transformação de conversão para tensor
transform = ToTensor()

# Cria o dataset com a transformação definida
dataset_train = COCODataset(root=img_folder_train, annFile=ann_file_train, transform=transform)
dataset_test = COCODataset(root=img_folder_teste, annFile=ann_file_teste, transform=transform)

# Função para salvar o dataset em disco
def save_dataset(dataset, file_path):
    data_list = []
    for img, boxes, labels, img_path in dataset:
        data_list.append((img, boxes, labels, img_path))
    torch.save(data_list, file_path)

# Salvar datasets de treino e teste
save_dataset(dataset_train, 'coco_train_dataset.pt')
save_dataset(dataset_test, 'coco_test_dataset.pt')
