# HandWriter
<h1>📚 STACKS</h1>

언어: Python <img src="https://img.shields.io/badge/Python-#3776AB?style=for-the-badge&logo=python&logoColor=white">
<br>

도구: 
<br>

협업툴:
<br>

# 데이터 변환 및 DataLoader 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = HandwritingDataset(train_data, transform=transform)
    test_dataset = HandwritingDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

batch size 64이며 가져오는 이미지 파일을 224*224 pixel로 사이즈를 조정하여 사용한다.

# ResNet-34
ResNet-34은 34개의 layer로 이루어진 심층 신경망 아키텍처 중 하나다. ResNet은 잔차 학습(residual learning)을 통해 네트워크를 훈련시키는 특징이 있어 더 깊은 네트워크를 효과적으로 학습할 수 있다. 이 구현에서는 torchvision에서 제공하는 미리 훈련된 ResNet-34 모델을 사용하고, 모델의 Fully Connected (FC) 레이어만 수정하여 새로운 클래스 수에 맞게 조정했다.

모델 정의:

    class HandwritingClassifier(nn.Module):
        def __init__(self, output_size):
            super(HandwritingClassifier, self).__init__()
            self.model = resnet34(pretrained=False)
            self.model.fc = nn.Linear(512, output_size)

        def forward(self, x):
            return self.model(x)

여기서 resnet34(pretrained=False)는 미리 훈련된 가중치를 사용하지 않는 ResNet-34 모델을 생성한다. 그리고 self.model.fc = nn.Linear(512, output_size)는 ResNet 모델의 마지막 Fully Connected 레이어를 새로운 레이어로 대체한다. 이 레이어는 512개의 입력 피처를 받아서 output_size의 출력을 내보낸다.

# 훈련 루프

훈련 루프에서는 이 모델이 사용되어 입력 이미지를 받아 예측을 수행하고, 손실을 계산하여 역전파를 수행하고 최적화를 진행한다.
훈련 루프 중 모델 사용 부분:

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
이 코드에서 model(inputs)는 ResNet 모델의 forward 함수를 호출하여 예측을 생성한다. 이것은 outputs에 저장되고, 이후에 손실 및 역전파가 이루어진다.

# 테스트 루프

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

이 코드에서 model(x)는 모델을 사용하여 이미지의 예측을 생성하고, 이를 시각화하기 위해 matplotlib을 사용한다.

# 성능 평가 지표 계산 및 출력
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 성능 평가 지표 계산
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_predictions)

# 결과 출력
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)

이 코드에서는 이미지 분류 모델에 대한 성능 평가 지표에 따른 값을 계산하고 나타내주는 결과물을 출력한다.
