import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Classes
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("best_waste_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # same ImageNet stats as training
                         std=[0.229, 0.224, 0.225])
])

# UI
st.title("♻️ Waste Classification AI")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    st.write(f"Prediction: **{classes[predicted.item()]}**")
    st.write(f"Confidence: **{confidence.item()*100:.2f}%**")