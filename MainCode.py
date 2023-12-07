import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet34
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로딩 및 전처리
data_path = "/content/drive/MyDrive/archive/"
image_folder = os.path.join(data_path, "Img")
csv_path = os.path.join(data_path, "english.csv")

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 이미지 파일 목록 생성
image_list = [os.path.join(image_folder, img) for img in df['image']]

# LabelEncoder를 사용하여 레이블을 숫자로 변환
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 데이터셋 분할
train_data, test_data = train_test_split(list(zip(image_list, df['label'])), test_size=0.1, random_state=42)

# 데이터셋 클래스 정의
class HandwritingDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')  # 흑백 이미지를 RGB로 변환
        if self.transform:
            img = self.transform(img)
        return img, label
    
# 데이터 변환 및 DataLoader 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = HandwritingDataset(train_data, transform=transform)
test_dataset = HandwritingDataset(test_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 정의
class HandwritingClassifier(nn.Module):
    def __init__(self, output_size):
        super(HandwritingClassifier, self).__init__()
        self.model = resnet34(pretrained=False)
        self.model.fc = nn.Linear(512, output_size)

    def forward(self, x):
        return self.model(x)

# 모델, 손실 함수, 옵티마이저 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HandwritingClassifier(output_size=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 훈련 루프
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.2f}%')

# 테스트 루프
model.eval()
with torch.no_grad():
    for i in range(10):  # 랜덤한 10개 이미지에 대한 예측 결과 시각화
        idx = np.random.randint(len(test_data))
        img_path, y = test_data[idx]

        # 이미지를 텐서로 변환
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        x = transform(img).unsqueeze(0).to(device)

        predict = model(x)
        predict = torch.argmax(predict)

        # label_encoder를 통해 클래스 이름으로 변환
        predicted_class = label_encoder.classes_[predict]
        actual_class = label_encoder.classes_[y]

        x = x.reshape((3, 224, 224)).to('cpu').numpy()
        plt.imshow(x.transpose(1, 2, 0))
        plt.title(f'Predicted: {predicted_class}, Actual: {actual_class}')
        plt.show()

print('Training and Testing finished!')