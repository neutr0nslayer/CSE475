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

import warnings
warnings.filterwarnings("ignore")


# Load the uploaded CSV files to inspect the data
train_image_paths = pd.read_csv('Dataset\\MURA-v1.1\\merged_train_image_labels.csv')
valid_image_paths = pd.read_csv('Dataset\\MURA-v1.1/merged_valid_image_labels.csv')
train_labeled_studies = pd.read_csv('Dataset\\MURA-v1.1\\merged_train_image_labels.csv')
valid_labeled_studies = pd.read_csv('Dataset\\MURA-v1.1/merged_valid_image_labels.csv')

# fill nulls in the 2nd column of each loaded DF with 1
for df in [train_image_paths, valid_image_paths, train_labeled_studies, valid_labeled_studies]:
    second_col = df.columns[1]
    df[second_col] = df[second_col].fillna(1)
    
# Display the first few rows of each file to understand their structure
# train_image_paths.head(), valid_image_paths.head(), train_labeled_studies.head(), valid_labeled_studies.head()
# train_image_paths.info()
# valid_image_paths.info()


# Custom Dataset class to handle images and labels
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

# Combine image paths and labels for training and validation
# Fix the dataset merging by ensuring the columns match properly
train_data = pd.DataFrame({
    'image_path': train_image_paths.iloc[:, 0],  # Use the first column for image paths
    'label': train_image_paths.iloc[:, 1]    # Use the second column for labels
})

valid_data = pd.DataFrame({
    'image_path': valid_image_paths.iloc[:, 0],  # Use the first column for image paths
    'label': valid_image_paths.iloc[:, 1]    # Use the second column for labels
})

# # Verify the structure after correction
# len(train_data), len(valid_data), train_data.head(), valid_data.head()


simclr_transform = transforms.Compose([
    transforms.Resize((246, 246)),
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

# simclr_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# Create Dataset instances for training and validation
train_dataset = XrayDataset(image_paths=train_data['image_path'].values, labels=train_data['label'].values, transform=simclr_transform)
valid_dataset = XrayDataset(image_paths=valid_data['image_path'].values, labels=valid_data['label'].values, transform=simclr_transform)

# Create DataLoader instances for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8)

# Verify the dataset and dataloader setup by checking a sample
# sample_image, sample_label = train_dataset[0]
# sample_image.size(), sample_label



# SimCLR model definition with fixed access to feature dimension
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

# Print the model to verify
simclr_model



# NT-Xent Loss Implementation

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Normalize the projections (L2 normalization)
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        # Compute cosine similarity between pairs
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)

        # Calculate NT-Xent loss (contrastive loss)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
        
        
# Set up the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# print(f"Using device: {device}")



# Move the model to the appropriate device (GPU/CPU)
simclr_model = simclr_model.to(device)

# Define the NT-Xent loss
criterion = NTXentLoss(temperature=0.5)

# Set up the optimizer
optimizer = torch.optim.Adam(simclr_model.parameters(), lr=1e-3)

# Verify the setup
simclr_model, criterion, optimizer



from tqdm import tqdm

# Updated Training Loop for SimCLR with Progress Bar

def train_simclr_with_early_stopping(model, train_loader, valid_loader, criterion, optimizer, device, epochs=50, patience=5):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, _ in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            images = images.to(device)
            images_1 = images
            images_2 = images.flip(0)

            optimizer.zero_grad()
            projections_1 = model(images_1)
            projections_2 = model(images_2)
            loss = criterion(projections_1, projections_2)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, _ in valid_loader:
                images = images.to(device)
                images_1 = images
                images_2 = images.flip(0)

                proj1 = model(images_1)
                proj2 = model(images_2)
                val_loss += criterion(proj1, proj2).item()
            avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

if __name__ == '__main__':
    # Place the training function call here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    trained_model = train_simclr_with_early_stopping(simclr_model, train_loader,valid_loader, criterion, optimizer, device, epochs=50, patience=5)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'simclr_model.pth')
    
    
