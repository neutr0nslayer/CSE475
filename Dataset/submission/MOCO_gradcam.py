import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === MoCo Encoder & Classifier Classes ===

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
    if 'state_dict' in state_dict:
        state_dict = {k.replace('module.', '').replace('encoder_q.', ''): v
                      for k, v in state_dict['state_dict'].items() if 'encoder_q' in k}
    else:
        state_dict = {k.replace('module.', '').replace('encoder_q.', ''): v
                      for k, v in state_dict.items() if 'encoder_q' in k}
    model.encoder_q.load_state_dict(state_dict, strict=False)
    return model


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


# === Grad-CAM ===

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
        cam = cam / np.max(cam + 1e-8)
        return cam


# === Visualization ===

def show_cam_on_image(img_path, cam, alpha=0.5):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_np = np.array(img) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    superimposed_img = heatmap * alpha + img_np

    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(superimposed_img, 0, 1))
    plt.axis('off')
    plt.title('MoCo Grad-CAM')
    plt.show()


# === Transforms ===

def get_transforms():
    return transforms.Compose([
        transforms.Resize((246, 246)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# === Run ===

def main():
    #MURA-v1.1/valid/XR_WRIST/patient11223/study1_positive/image2.png
    #MURA-v1.1/valid/XR_WRIST/patient11205/study1_positive/image2.png
    dir_img = "Dataset/"
    image_path = "MURA-v1.1/valid/XR_WRIST/patient11205/study1_positive/image2.png"
    image_path = os.path.join(dir_img, image_path)
    encoder_path = "best_moco_model.pth"
    classifier_path = "best_moco_classifier.pth"

    # Load MoCo encoder and classifier
    moco_encoder = load_moco_encoder(encoder_path)
    model = MoCoClassifier(moco_encoder, num_classes=2, freeze_encoder=False)
    model.load_state_dict(torch.load(classifier_path, map_location=device))
    model.to(device)

    # Prepare image
    img = Image.open(image_path).convert("RGB")
    transform = get_transforms()
    input_tensor = transform(img)

    # Last ResNet block
    target_layer = model.encoder.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # Generate CAM
    cam = grad_cam.generate(input_tensor)
    show_cam_on_image(image_path, cam)


if __name__ == "__main__":
    main()
