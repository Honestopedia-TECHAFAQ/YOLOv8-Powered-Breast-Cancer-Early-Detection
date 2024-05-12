import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ["cancerous", "non_cancerous"]
        self.image_paths = []
        self.labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            for image_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_name))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}
class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8, self).__init__()

    def forward(self, x):
        return x
class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, predictions, targets):
        return loss
num_epochs = 10
batch_size = 32
learning_rate = 0.001
train_dataset = CustomDataset(data_dir='path_to_train_data', transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset(data_dir='path_to_val_data', transform=data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = YOLOv8(num_classes=1) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = YOLOLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
