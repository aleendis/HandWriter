# HandWriter

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
