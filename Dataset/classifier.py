import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the uploaded CSV files to inspect the data
train_image_paths = pd.read_csv('Dataset\\MURA-v1.1\\merged_train_image_labels.csv')
valid_image_paths = pd.read_csv('Dataset\\MURA-v1.1/merged_valid_image_labels.csv')
train_labeled_studies = pd.read_csv('Dataset\\MURA-v1.1\\merged_train_image_labels.csv')
valid_labeled_studies = pd.read_csv('Dataset\\MURA-v1.1/merged_valid_image_labels.csv')

# fill nulls in the 2nd column of each loaded DF with 1
for df in [train_image_paths, valid_image_paths, train_labeled_studies, valid_labeled_studies]:
    second_col = df.columns[1]
    df[second_col] = df[second_col].fillna(1)

base_dir = 'Dataset/'

# Update the Dataset class to include the base directory in image paths
class XrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = [os.path.join(base_dir, path) for path in image_paths]  # Prepend the base directory to paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = int(self.labels[idx])  # Ensure the label is an integer (0 or 1)

        # Open image
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

train_data = pd.DataFrame({
    'image_path': train_image_paths.iloc[:, 0],  # Use the first column for image paths
    'label': train_image_paths.iloc[:, 1]    # Use the second column for labels
})

valid_data = pd.DataFrame({
    'image_path': valid_image_paths.iloc[:, 0],  # Use the first column for image paths
    'label': valid_image_paths.iloc[:, 1]    # Use the second column for labels
})

# Verify the structure after correction
#len(train_data), len(valid_data), train_data.head(), valid_data.head()


simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4,0.4,0.4,0.1)],
        p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

train_dataset = XrayDataset(image_paths=train_data['image_path'].values, labels=train_data['label'].values, transform=simclr_transform)
valid_dataset = XrayDataset(image_paths=valid_data['image_path'].values, labels=valid_data['label'].values, transform=simclr_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8)

class SimCLR(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLR, self).__init__()
        self.base_model = base_model
        # Remove the fully connected layer (fc)
        self.base_model.fc = nn.Identity()
        # Use the known feature dimension of ResNet-50 (2048)
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim),
        )

    def forward(self, x):
        features = self.base_model(x)
        projections = self.projection_head(features)
        return projections
    
# Load a pre-trained ResNet model
resnet_model = models.resnet50(pretrained=True)

# Create the SimCLR model
simclr_model = SimCLR(resnet_model)



class SimCLRClassifier(nn.Module):
    def __init__(self, simclr_model, num_classes=2):
        super(SimCLRClassifier, self).__init__()
        self.encoder = simclr_model.base_model  # ResNet without the fc layer
        self.classifier = nn.Linear(2048, num_classes)  # Add classification layer

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
simclr_model.load_state_dict(torch.load('E:\\CSE475\\simclr_model.pth'))
classifier_model = SimCLRClassifier(simclr_model)
classifier_model = classifier_model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-4)


def train_classifier(model, train_loader, valid_loader, criterion, optimizer, device, epochs=5):
    model = model.to(device)
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(train_loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation Phase ===
        model.eval()
        total_valid_loss = 0
        all_preds = []
        all_labels = []

        valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch")
        with torch.no_grad():
            for images, labels in valid_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_valid_loss:.4f}")
        

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Abnormal"]))
    
    # === Plot Losses ===
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model



if __name__ == '__main__':
    trained_classifier = train_classifier(
        classifier_model, train_loader, valid_loader, criterion, optimizer, device, epochs=5
    )
    
    torch.save(trained_classifier.state_dict(), 'simclr_classifier_model.pth')



