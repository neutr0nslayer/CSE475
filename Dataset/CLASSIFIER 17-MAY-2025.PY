import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
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
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((246, 246)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def train_classifier(model, train_loader, valid_loader, criterion, optimizer, device, epochs=50, patience=5, log_path='training_log.txt', plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)

    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)

    model = model.to(device)
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    best_val_loss = float('inf')
    patience_counter = 0
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    plt.close()

    # ROC AUC Curve
    try:
        model.eval()
        all_probs = []
        with torch.no_grad():
            for images, _ in valid_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1: Abnormal
                all_probs.extend(probs.cpu().numpy())

        auc_score = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'roc_auc_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Failed to plot ROC AUC: {e}")

    # Training Curves
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
    plt.savefig(os.path.join(plot_dir, 'training_curves.png'))
    plt.close()

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

    return model


def main():
    base_dir = 'Dataset/'
    base_dir1 = '.'
    train_csv = 'Dataset/balanced_train_image_labels.csv'
    valid_csv = os.path.join(base_dir, 'MURA-v1.1', 'merged_valid_image_labels.csv')
    pretrained_simclr_path = 'best_simclr_model.pth'
    batch_size = 32
    num_workers = 8
    num_classes = 2
    learning_rate = 1e-4
    num_epochs = 50
    patience = 50
    log_file = 'training_log.txt'
    plot_dir = 'plots'

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

    transform = get_transforms()
    train_dataset = XrayDataset(train_data['image_path'].values, train_data['label'].values, transform, base_dir1)
    valid_dataset = XrayDataset(valid_data['image_path'].values, valid_data['label'].values, transform, base_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
        patience=patience,
        log_path=log_file,
        plot_dir=plot_dir
    )

    os.makedirs('trained_model', exist_ok=True)
    torch.save(trained_model.state_dict(), 'trained_model/18simclr_classifier_model.pth')


if __name__ == '__main__':
    main()
