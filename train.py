import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import FaceMaskDataset
from sklearn.metrics import accuracy_score
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device)


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


train_dataset = FaceMaskDataset("labels.csv", "data/train", transform=transform)
val_dataset = FaceMaskDataset("labels.csv", "data/val", transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_preds = []
    train_targets = []

    for images, labels in train_loader:
        print("Bir batch yüklendi...")
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_targets, train_preds)


    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_targets, val_preds)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
    print("-" * 50)

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet18_mask_classifier.pth")
print("The model is saved.")
