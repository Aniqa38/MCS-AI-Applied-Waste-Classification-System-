import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load classes (same order as training)
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("best_waste_model.pth"))
model.eval()

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # same ImageNet stats as training
                         std=[0.229, 0.224, 0.225])
])

# Load your test image
img_path = "test.jpg"  # put any image here
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Prediction:", classes[predicted.item()])