# === [1] All imports and setup ===
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === [2] Dataset ===
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


# === [3] MoCo Encoder Loader ===
class MoCoEncoder(nn.Module):
    def __init__(self, dim=128):
        super(MoCoEncoder, self).__init__()
        base_encoder = models.resnet50
        self.encoder_q = base_encoder(pretrained=False)
        self.encoder_q.fc = nn.Identity()
        self.projector_q = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

    def forward(self, x):
        features = self.encoder_q(x)
        projections = self.projector_q(features)
        return features, projections


def load_moco_encoder(path):
    model = MoCoEncoder()
    state_dict = torch.load(path, map_location=device)
    # Handle both .pth and .tar files
    if 'state_dict' in state_dict:
        state_dict = {k.replace('module.', '').replace('encoder_q.', ''): v
                      for k, v in state_dict['state_dict'].items() if 'encoder_q' in k}
    else:
        state_dict = {k.replace('module.', '').replace('encoder_q.', ''): v
                      for k, v in state_dict.items() if 'encoder_q' in k}
    model.encoder_q.load_state_dict(state_dict, strict=False)
    return model


# === [4] Classifier ===
class MoCoClassifier(nn.Module):
    def __init__(self, moco_model, num_classes=2, freeze_encoder=False):
        super(MoCoClassifier, self).__init__()
        self.encoder = moco_model.encoder_q
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

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
            torch.save(model.state_dict(), 'best_moco_classifier.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Abnormal"]))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    plt.close()

    try:
        model.eval()
        all_probs = []
        with torch.no_grad():
            for images, _ in valid_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
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
        plt.savefig(os.path.join(plot_dir, 'moco_roc_auc_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Failed to plot ROC AUC: {e}")

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
    plt.savefig(os.path.join(plot_dir, 'moco_training_curves.png'))
    plt.close()

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    return model


def main():
    # Paths
    base_dir = 'Dataset/'
    base_dir1 = '.'
    train_csv = os.path.join(base_dir, 'balanced_train_image_labels.csv')
    valid_csv = os.path.join(base_dir, 'MURA-v1.1', 'merged_valid_image_labels.csv')
    pretrained_moco_path = 'best_moco_model.pth'
    batch_size = 32
    num_classes = 2
    learning_rate = 1e-4
    num_epochs = 50
    patience = 5
    log_file = 'moco_training_log.txt'
    plot_dir = 'plots'

    # Load CSVs
    train_df = pd.read_csv(train_csv).fillna(1)
    valid_df = pd.read_csv(valid_csv).fillna(1)

    train_dataset = XrayDataset(train_df.iloc[:, 0].values, train_df.iloc[:, 1].values, get_transforms(), base_dir1)
    valid_dataset = XrayDataset(valid_df.iloc[:, 0].values, valid_df.iloc[:, 1].values, get_transforms(), base_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load pretrained MoCo encoder
    moco_encoder = load_moco_encoder(pretrained_moco_path)

    # Create classifier model
    classifier_model = MoCoClassifier(moco_encoder, num_classes).to(device)

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)

    # Train
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
    torch.save(trained_model.state_dict(), 'trained_model/moco_classifier_model.pth')


if __name__ == '__main__':
    main()