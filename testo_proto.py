import os
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random

# Кастомные трансформации (crop, resize, jitter, rotate, normalize)
class Transforms:
    def __init__(self):
        pass

    def rotate(self, image, landmarks, angle=10):
        angle = random.uniform(-angle, angle)
        w, h = image.size
        image = TF.rotate(image, angle)

        # центр картинки
        cx, cy = w / 2.0, h / 2.0

        # переносим в систему координат центра
        x = landmarks[:, 0] - cx
        y = landmarks[:, 1] - cy

        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a

        landmarks[:, 0] = x_new + cx
        landmarks[:, 1] = y_new + cy

        return image, landmarks


    def resize(self, image, landmarks, size):
        w, h = image.size
        image = image.resize(size)
        landmarks[:, 0] *= size[0] / w
        landmarks[:, 1] *= size[1] / h
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left, top, right, bottom = crops
        image = TF.crop(image, top, left, bottom - top, right - left)
        landmarks[:, 0] -= left
        landmarks[:, 1] -= top
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks)
        image = TF.to_grayscale(image)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        landmarks = torch.tensor(landmarks.flatten() / 224, dtype=torch.float32)  # Нормализуем в [0,1]
        return image, landmarks

# Dataset класс (парсит XML из iBUG 300-W)
class FaceLandmarksDataset(Dataset):
    def __init__(self, root_dir, xml_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        tree = ET.parse(xml_file)
        root = tree.getroot()
        self.image_data = []
        for image in root[2]:
            filename = image.attrib['file']
            boxes = []
            landmarks = []
            for box in image:
                boxes.append([int(box.attrib['left']), int(box.attrib['top']), 
                              int(box.attrib['left']) + int(box.attrib['width']), 
                              int(box.attrib['top']) + int(box.attrib['height'])])
                landmark = []
                for part in box:
                    landmark.append([int(part.attrib['x']), int(part.attrib['y'])])
                landmarks.append(landmark)
            self.image_data.append({'file': filename, 'boxes': boxes, 'landmarks': landmarks})

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        data = self.image_data[idx]
        img_path = os.path.join(self.root_dir, data['file'])
        image = np.array(Image.open(img_path).convert('RGB'))
        landmarks = np.array(data['landmarks'][0])  # Берем первую аннотацию (одно лицо)
        landmarks = landmarks.astype(np.float32)  # Add this line to fix the dtype issue
        crops = data['boxes'][0]
        if self.transform:
            image, landmarks = self.transform(image, landmarks, crops)
        return image, landmarks

# Модель: ResNet18 с модификациями (1 канал вход, 136 выходов)
class Network(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Функция для расчёта fWHR из landmarks (68 точек)
def calculate_fwhr(landmarks):
    landmarks = landmarks.reshape(68, 2) * 224  # Денормализуем
    # Ширина: расстояние между zygions (точки 0 и 16)
    width = np.linalg.norm(landmarks[16] - landmarks[0])
    # Высота: от брови (средняя, точка 27) до верхней губы (точка 33) или chin (8), но стандарт - bizygomatic width / upper face height
    height = np.linalg.norm(landmarks[8] - landmarks[27])  # Пример: от chin до glabella
    fwhr = width / height if height != 0 else 0
    return fwhr

# Тренировка (упрощённая)
def train_model():
    dataset = FaceLandmarksDataset(root_dir='ibug_300W_large_face_landmark_dataset', 
                                   xml_file='ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml', 
                                   transform=Transforms())
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, valid_size],
        generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Using device:', device)
    model = Network().to(device)
    print(f'Using device: {device}')  # For debugging

    model = Network().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_loss = float('inf')
    for epoch in range(10):  # 10 эпох
        model.train()
        running_loss = 0.0
        for images, landmarks in train_loader:
            images, landmarks = images.to(device), landmarks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

        # Валидация (упрощённо)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, landmarks in valid_loader:
                images, landmarks = images.to(device), landmarks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, landmarks).item()
        val_loss /= len(valid_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'testo_model.pth')
        print(f'Val Loss: {val_loss}')

# Инференс: Анализ фото
def analyze_photo(photo_path):
    model = Network()
    model.load_state_dict(torch.load('testo_model.pth'))
    model.eval()

    image = np.array(Image.open(photo_path).convert('RGB'))
    # Предполагаем, что лицо уже обрезано; в реале добавь face detection (e.g., MTCNN)
    crops = [0, 0, image.shape[1], image.shape[0]]  # Dummy crop
    transform = Transforms()
    image, _ = transform(image, np.zeros((68, 2)), crops)  # Игнорируем landmarks для инференса
    image = image.unsqueeze(0)  # Batch dim

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = Network().to(device)
    state = torch.load('testo_model.pth', map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    image = image.to(device)
    with torch.no_grad():
        landmarks = model(image).cpu().numpy()


    fwhr = calculate_fwhr(landmarks)
    if fwhr > 2.0:
        level = 'Высокий'
    elif fwhr > 1.8:
        level = 'Средний'
    else:
        level = 'Низкий'

    print(f'fWHR: {fwhr:.2f}, Уровень тестостерона (оценка): {level}')

    # Визуализация
    landmarks = landmarks.reshape(68, 2) * 224
    plt.imshow(np.array(Image.open(photo_path).resize((224, 224)).convert('L')), cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='r')
    plt.show()

# Запуск
if __name__ == '__main__':
    # train_model()  # Раскомментируй для тренировки (нужен dataset)
    analyze_photo('path/to/your/photo.jpg')  # Тестируй на своём фото