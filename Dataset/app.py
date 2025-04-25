# app.py

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import gradio as gr

# === Model Definitions ===
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
        logits = self.classifier(features)
        return logits

# === Load Model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
simclr_model = SimCLR(base_resnet)

classifier_model = SimCLRClassifier(simclr_model)
classifier_model.load_state_dict(torch.load('simclr_classifier_model.pth', map_location=device))
classifier_model = classifier_model.to(device)
classifier_model.eval()

# === Transform ===
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


# === Prediction Function ===
def predict_image(image):
    image = image.convert('RGB')
    image = simclr_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier_model(image)
        _, preds = torch.max(outputs, 1)

    classes = ["Normal", "Abnormal"]
    return classes[preds.item()]

# === Gradio Interface ===
def main():
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload X-ray Image"),
        outputs=gr.Label(label="Prediction"),
        title="X-ray Diagnosis Predictor",
        description="Upload an X-ray image and the model will predict if it is Normal or Abnormal.",
        theme="default"
    )
    interface.launch()

if __name__ == "__main__":
    main()
