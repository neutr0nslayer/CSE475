import os
import time
import warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# Custom Dataset class
class XrayDataset(Dataset):
    def __init__(self, image_paths, labels, base_dir, transform=None):
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


# SimCLR Model
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


# NT-Xent Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(similarity_matrix, labels)



# Training function
def train_simclr(model, train_loader, criterion, optimizer, device, epochs=50, patience=5):
    model.train()
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    avg_loss = 0.0
    
    for epoch in range(epochs):
        print(f"Patience Counter: {patience_counter}, Best Loss: {best_loss} , Avg Loss: {avg_loss}")
        running_loss = 0.0
        model.train()
        for images, _ in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            images = images.to(device)
            images_1 = images
            images_2 = images.flip(0)

            optimizer.zero_grad()
            proj_1 = model(images_1)
            proj_2 = model(images_2)
            loss = criterion(proj_1, proj_2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, ")
        print()

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_simclr_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        

    return model, train_losses



    
    
# Main function
def main(
    train_csv='Dataset/balanced_train_image_labels.csv',
    valid_csv='Dataset/MURA-v1.1/merged_valid_image_labels.csv',
    base_dir='.',
    batch_size=32, # 16 4
    num_workers = min(os.cpu_count(), 10),  
    lr=1e-3,
    epochs=50,
    patience=5,
    projection_dim=128,
    temperature=0.5,
    model_save_path='trained_model/simclr_model.pth'
):
    # Load CSVs
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    for df in [train_df, valid_df]:
        df[df.columns[1]] = df[df.columns[1]].fillna(1)

    train_data = pd.DataFrame({'image_path': train_df.iloc[:, 0], 'label': train_df.iloc[:, 1]})
    valid_data = pd.DataFrame({'image_path': valid_df.iloc[:, 0], 'label': valid_df.iloc[:, 1]})

    transform = transforms.Compose([
        transforms.Resize((246, 246)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = XrayDataset(train_data['image_path'].values, train_data['label'].values, base_dir, transform)
    valid_dataset = XrayDataset(valid_data['image_path'].values, valid_data['label'].values, base_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}|| Batch size: {batch_size}|| Num workers: {num_workers}")
    
    resnet_base = models.resnet50(pretrained=True)
    simclr_model = SimCLR(resnet_base, projection_dim=projection_dim).to(device)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(simclr_model.parameters(), lr=lr)

    trained_model, train_losses= train_simclr(simclr_model, train_loader, criterion, optimizer, device, epochs)
    # def train_simclr(model, train_loader, criterion, optimizer, device, epochs=50):
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()


# Only run main if executed directly
if __name__ == '__main__':
    main()
