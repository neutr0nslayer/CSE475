import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, base_dir=''):
        self.image_paths = [os.path.join(base_dir, path) for path in image_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = int(self.labels[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SimCLR(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLR, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim),
        )

    def forward(self, x):
        features = self.base_model(x)
        projections = self.projection_head(features)
        return projections


class SimCLRClassifier(nn.Module):
    def __init__(self, simclr_model, num_classes=2):
        super(SimCLRClassifier, self).__init__()
        self.encoder = simclr_model.base_model
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((246, 246)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def train_classifier(model, train_loader, valid_loader, criterion, optimizer, device, epochs=50, patience=5):
    model = model.to(device)
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        total_valid_loss = 0
        correct_val = 0
        total_val = 0
        all_preds, all_labels = [], []

        valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", unit="batch")
        with torch.no_grad():
            for images, labels in valid_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_valid_loss = total_valid_loss / len(valid_loader)
        val_acc = correct_val / total_val
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(val_acc)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_valid_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            torch.save(model.state_dict(), 'best_simclr_classifier.pth') 
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Abnormal"]))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(valid_accuracies, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model


def main():
    # ==== Configurable Variables ====
    base_dir = 'Dataset/'
    train_csv = os.path.join(base_dir, 'MURA-v1.1', 'merged_train_image_labels.csv')
    valid_csv = os.path.join(base_dir, 'MURA-v1.1', 'merged_valid_image_labels.csv')
    pretrained_simclr_path = 'E:\\CSE475\\simclr_model.pth'
    batch_size = 32
    num_workers = 8
    num_classes = 2
    learning_rate = 1e-4
    num_epochs = 50
    patience = 5

    # ==== Load CSVs and Prepare DataFrames ====
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    second_col = train_df.columns[1]
    train_df[second_col] = train_df[second_col].fillna(1)
    valid_df[second_col] = valid_df[second_col].fillna(1)

    train_data = pd.DataFrame({
        'image_path': train_df.iloc[:, 0],
        'label': train_df.iloc[:, 1]
    })

    valid_data = pd.DataFrame({
        'image_path': valid_df.iloc[:, 0],
        'label': valid_df.iloc[:, 1]
    })

    # ==== Prepare Datasets & Loaders ====
    transform = get_transforms()
    train_dataset = XrayDataset(train_data['image_path'].values, train_data['label'].values, transform, base_dir)
    valid_dataset = XrayDataset(valid_data['image_path'].values, valid_data['label'].values, transform, base_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ==== Load Models ====
    resnet = models.resnet50(pretrained=True)
    simclr = SimCLR(resnet)
    simclr.load_state_dict(torch.load(pretrained_simclr_path, map_location=device))
    classifier_model = SimCLRClassifier(simclr, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)

    trained_model = train_classifier(
        classifier_model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        epochs=num_epochs,
        patience=patience
    )

    torch.save(trained_model.state_dict(), 'simclr_classifier_model.pth')


if __name__ == '__main__':
    main()
