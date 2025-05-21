import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- SimCLR + Classifier Architecture (same as your model) ---

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


# --- Grad-CAM Class ---

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.unsqueeze(0).to(device)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


# --- Visualization ---

def show_cam_on_image(img_path, cam, alpha=0.5):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_np = np.array(img) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    superimposed_img = heatmap * alpha + img_np

    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(superimposed_img, 0, 1))
    plt.axis('off')
    plt.title('Grad-CAM')
    plt.show()


# --- Preprocessing ---

def get_transforms():
    return transforms.Compose([
        transforms.Resize((246, 246)),
        transforms.CenterCrop(224),  # For GradCAM we avoid aggressive augmentations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# --- Load and Run ---

def main():
    # Path setup
    #MURA-v1.1/valid/XR_WRIST/patient11223/study1_positive/image2.png
    #MURA-v1.1/valid/XR_WRIST/patient11205/study1_positive/image2.png
    dir_img = "Dataset/"
    image_path = "MURA-v1.1/valid/XR_WRIST/patient11223/study1_positive/image2.png"
    image_path = os.path.join(dir_img, image_path)
    classifier_weights_path = "best_simclr_classifier.pth"
    simclr_pretrain_path = "best_simclr_model.pth"

    # Load model
    base_model = models.resnet50(pretrained=True)
    simclr = SimCLR(base_model)
    simclr.load_state_dict(torch.load(simclr_pretrain_path, map_location=device))

    model = SimCLRClassifier(simclr, num_classes=2)
    model.load_state_dict(torch.load(classifier_weights_path, map_location=device))
    model.to(device)

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    transform = get_transforms()
    input_tensor = transform(img)

    # Target layer: last conv layer of ResNet
    target_layer = model.encoder.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # Generate and show CAM
    cam = grad_cam.generate(input_tensor)
    show_cam_on_image(image_path, cam)


if __name__ == "__main__":
    main()
