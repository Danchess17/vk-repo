import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. Загрузка предварительно обученной модели (ResNet50)
model = models.resnet50(pretrained=True)
# Удаляем последние слои (classifier)
model = nn.Sequential(*list(model.children())[:-1])  # Remove last layer (avgpool and fc)
model.eval()

# 2. Предобработка изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    """Извлекает признаки из изображения с помощью ResNet50."""
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0) # Добавляем размерность батча
    with torch.no_grad():
        features = model(image)
    return features.flatten().numpy()

# 3. Создание базы данных эмбеддингов известных логотипов
def create_logo_database(logo_dir):
    """Создает базу данных эмбеддингов логотипов."""
    logo_embeddings = {}
    for filename in os.listdir(logo_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(logo_dir, filename)
            logo_embeddings[filename] = extract_features(image_path)
    return logo_embeddings

# 4. Распознавание логотипа
def recognize_logo(input_image_path, logo_embeddings, threshold=0.65):
    """Определяет, является ли изображение логотипом."""
    input_embedding = extract_features(input_image_path)
    max_similarity = 0
    best_match = None

    for logo_name, logo_embedding in logo_embeddings.items():
        similarity = cosine_similarity([input_embedding], [logo_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = logo_name

    if max_similarity > threshold:
        return True, best_match, max_similarity
    else:
        return False, None, None


# 1.  Создайте папку "logos" и поместите туда несколько изображений
#     известных логотипов искомой организации (например, Coca-Cola).

# 2.  Обучите модель
logo_database = create_logo_database("logos")

# 3.  Запустите распознавание
input_image = "images/2.jpg"  # Замените на путь к вашему тестовому изображению
is_logo, best_match, similarity_score = recognize_logo(input_image, logo_database)

if is_logo:
    print(f"Изображение является логотипом. Наиболее похоже на: {best_match}, score: {similarity_score:.2f}")
else:
    print("Изображение не является логотипом.")
