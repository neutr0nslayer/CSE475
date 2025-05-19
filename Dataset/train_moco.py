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
import copy

warnings.filterwarnings("ignore")


# X-ray Dataset
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


# MoCo Model
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=4096, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # Online encoder
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_q.fc = nn.Identity()
        self.projector_q = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        # Momentum encoder
        self.encoder_k = base_encoder(num_classes=dim)
        self.encoder_k.fc = nn.Identity()
        self.projector_k = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        # Initialize key encoder with query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.projector_q(self.encoder_q(im_q))
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = self.projector_k(self.encoder_k(im_k))
            k = nn.functional.normalize(k, dim=1)

        # Compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (N,1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # (N,K)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        self.dequeue_and_enqueue(k)
        return logits, labels


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [tensor]
    return torch.cat(tensors_gather, dim=0)


def train_moco(model, dataloader, optimizer, device, epochs=50, patience=5):
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, _ in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            images = images.to(device)
            images_q = images
            images_k = images.flip(0)

            logits, labels = model(images_q, images_k)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_moco_model.pth')
            print("Model improved and saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model, loss_history


# Main runner
def main(
    train_csv='Dataset/balanced_train_image_labels.csv',
    base_dir='.',
    batch_size=32,
    num_workers=min(os.cpu_count(), 10),
    lr=1e-3,
    epochs=50,
    patience=5,
    projection_dim=128,
    temperature=0.07,
    K=4096,
    momentum=0.999
):
    # Load data
    train_df = pd.read_csv(train_csv)
    train_df[train_df.columns[1]] = train_df[train_df.columns[1]].fillna(1)
    train_data = pd.DataFrame({'image_path': train_df.iloc[:, 0], 'label': train_df.iloc[:, 1]})

    transform = transforms.Compose([
        transforms.Resize((246, 246)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = XrayDataset(train_data['image_path'].values, train_data['label'].values, base_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True  )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resnet_base = models.resnet50
    moco_model = MoCo(resnet_base, dim=projection_dim, K=K, m=momentum, T=temperature).to(device)
    optimizer = torch.optim.Adam(moco_model.parameters(), lr=lr)

    trained_model, losses = train_moco(moco_model, train_loader, optimizer, device, epochs, patience)

    
    plt.plot(losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MoCo Training Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
