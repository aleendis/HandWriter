# HandWriter

<h1>📚 STACKS</h1>

언어: <img src="https://img.shields.io/badge/Python-#3776AB?style=for-the-badge&logo=python&logoColor=white">
<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>Python</title><path d="M14.25.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.77l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.06H3.17l-.21-.03-.28-.07-.32-.12-.35-.18-.36-.26-.36-.36-.35-.46-.32-.59-.28-.73-.21-.88-.14-1.05-.05-1.23.06-1.22.16-1.04.24-.87.32-.71.36-.57.4-.44.42-.33.42-.24.4-.16.36-.1.32-.05.24-.01h.16l.06.01h8.16v-.83H6.18l-.01-2.75-.02-.37.05-.34.11-.31.17-.28.25-.26.31-.23.38-.2.44-.18.51-.15.58-.12.64-.1.71-.06.77-.04.84-.02 1.27.05zm-6.3 1.98l-.23.33-.08.41.08.41.23.34.33.22.41.09.41-.09.33-.22.23-.34.08-.41-.08-.41-.23-.33-.33-.22-.41-.09-.41.09zm13.09 3.95l.28.06.32.12.35.18.36.27.36.35.35.47.32.59.28.73.21.88.14 1.04.05 1.23-.06 1.23-.16 1.04-.24.86-.32.71-.36.57-.4.45-.42.33-.42.24-.4.16-.36.09-.32.05-.24.02-.16-.01h-8.22v.82h5.84l.01 2.76.02.36-.05.34-.11.31-.17.29-.25.25-.31.24-.38.2-.44.17-.51.15-.58.13-.64.09-.71.07-.77.04-.84.01-1.27-.04-1.07-.14-.9-.2-.73-.25-.59-.3-.45-.33-.34-.34-.25-.34-.16-.33-.1-.3-.04-.25-.02-.2.01-.13v-5.34l.05-.64.13-.54.21-.46.26-.38.3-.32.33-.24.35-.2.35-.14.33-.1.3-.06.26-.04.21-.02.13-.01h5.84l.69-.05.59-.14.5-.21.41-.28.33-.32.27-.35.2-.36.15-.36.1-.35.07-.32.04-.28.02-.21V6.07h2.09l.14.01zm-6.47 14.25l-.23.33-.08.41.08.41.23.33.33.23.41.08.41-.08.33-.23.23-.33.08-.41-.08-.41-.23-.33-.33-.23-.41-.08-.41.08z"/></svg>
<br>

도구: <img src="https://img.shields.io/badge/Pytorch-#EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
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
