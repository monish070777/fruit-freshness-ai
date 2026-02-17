import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Fruit Freshness Detector", layout="centered")

st.title("🍎 Fruit Freshness Detection AI")
st.write("Upload a fruit image and the AI will detect whether it is **Fresh or Rotten**")

# -----------------------------
# Classes
# -----------------------------
classes = [
    'freshapples', 'freshbanana', 'freshoranges',
    'rottenapples', 'rottenbanana', 'rottenoranges'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Load Model (cached so reload fast)
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.load_state_dict(torch.load("resnet18fruit_V001.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict(img):
    img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs,1)

    idx = predicted.item()

    if idx >= len(classes):
        return "Unknown", 0.0, "Unknown"

    label = classes[idx]
    confidence = confidence.item()*100

    # Extract freshness
    if "fresh" in label:
        freshness = "Fresh"
    else:
        freshness = "Rotten"

    # Extract fruit name
    fruit_name = label.replace("fresh","").replace("rotten","")

    return fruit_name.capitalize(), confidence, freshness

# -----------------------------
# Upload UI
# -----------------------------
uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Freshness"):
        fruit, conf, freshness = predict(image)

        st.subheader("Prediction Result")

        if freshness == "Fresh":
            st.success(f"{fruit} is FRESH ✅")
        else:
            st.error(f"{fruit} is ROTTEN ❌")

        st.write(f"Confidence: **{conf:.2f}%**")