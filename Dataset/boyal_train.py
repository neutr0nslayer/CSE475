import os
import copy
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
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")

# BYOL Transform for two views
class BYOLTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

# Dataset class
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
            image1, image2 = self.transform(image)
        return image1, image2, label

# MLP Head for BYOL
class MLPHead(nn.Module):
    def __init__(self, in_dim=2048, projection_dim=256, hidden_dim=4096):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)

# BYOL Model
class BYOL(nn.Module):
    def __init__(self, backbone, projection_dim=256):
        super(BYOL, self).__init__()
        self.online_encoder = nn.Sequential(
            backbone,
            MLPHead(in_dim=2048, projection_dim=projection_dim)
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = MLPHead(in_dim=projection_dim, projection_dim=projection_dim)
        self._freeze_target_encoder()

    def _freeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        online_proj1 = self.online_encoder(x1)
        online_proj2 = self.online_encoder(x2)

        pred1 = self.predictor(online_proj1)
        pred2 = self.predictor(online_proj2)

        with torch.no_grad():
            target_proj1 = self.target_encoder(x1)
            target_proj2 = self.target_encoder(x2)

        return pred1, pred2, target_proj1.detach(), target_proj2.detach()

    def update_target_encoder(self, tau=0.996):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

# BYOL Loss

def byol_loss_fn(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()

# BYOL Training with AMP
def train_byol(model, train_loader, optimizer, device, epochs=100, tau=0.996, patience=5):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(epochs):
        running_loss = 0.0
        for (images1, images2, _) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images1, images2 = images1.to(device), images2.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                pred1, pred2, target1, target2 = model(images1, images2)
                loss = byol_loss_fn(pred1, target2) + byol_loss_fn(pred2, target1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_target_encoder(tau=tau)
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_byol_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model, train_losses

# Main function
def main(
    train_csv='Dataset/balanced_train_image_labels.csv',
    base_dir='.',
    batch_size=32,
    num_workers=min(os.cpu_count(), 10),
    lr=1e-3,
    epochs=20,
    patience=5,
    projection_dim=256,
    model_save_path='trained_model/byol_model.pth'
):
    train_df = pd.read_csv(train_csv)
    train_df[train_df.columns[1]] = train_df[train_df.columns[1]].fillna(1)
    train_data = pd.DataFrame({'image_path': train_df.iloc[:, 0], 'label': train_df.iloc[:, 1]})

    base_transform = transforms.Compose([
        transforms.Resize((246, 246)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = BYOLTransform(base_transform)
    train_dataset = XrayDataset(train_data['image_path'].values, train_data['label'].values, base_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} || Batch size: {batch_size} || Num workers: {num_workers}")

    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()

    byol_model = BYOL(resnet, projection_dim=projection_dim).to(device)
    optimizer = torch.optim.Adam(byol_model.parameters(), lr=lr)

    trained_model, train_losses = train_byol(byol_model, train_loader, optimizer, device, epochs, patience=patience)

    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BYOL Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('byol_loss_plot.png', dpi=300)
    print("Training loss plot saved to byol_loss_plot.png")

if __name__ == '__main__':
    main()
