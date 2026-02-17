
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import os
import gdown

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Fruit Freshness AI", page_icon="🍎", layout="centered")

# --------------------------------------------------
# DOWNLOAD MODEL (FIRST RUN ONLY)
# --------------------------------------------------
MODEL_PATH = "resnet18fruit_V001.pth"
FILE_ID = "13_u1jqh0jxxzmy3wegO7NFv8mmtr6xgw"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model... please wait ⏳ (first run only)"):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# --------------------------------------------------
# UI STYLE
# --------------------------------------------------
st.markdown("""
<style>
.main-title{text-align:center;font-size:38px;font-weight:700;color:#2E7D32;}
.sub{text-align:center;color:gray;margin-bottom:25px;}
.card{padding:18px;border-radius:18px;background:#f7f7f7;box-shadow:0 4px 15px rgba(0,0,0,0.08);}
.result-good{padding:20px;border-radius:15px;background:#e8f5e9;color:#1b5e20;font-size:24px;text-align:center;font-weight:700;}
.result-bad{padding:20px;border-radius:15px;background:#ffebee;color:#b71c1c;font-size:24px;text-align:center;font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🍎 Fruit Freshness Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Take photo and detect Fresh or Rotten</div>", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL LOAD
# --------------------------------------------------
classes = ['freshapples','freshbanana','freshoranges','rottenapples','rottenbanana','rottenoranges']
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    state_dict = torch.load(MODEL_PATH, map_location=device)

    new_state_dict={}
    for k,v in state_dict.items():
        new_state_dict[k.replace("module.","")] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

model = load_model()

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def predict(img):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs,1)

    idx = predicted.item()
    if idx >= len(classes):
        return "Unknown",0,"Unknown"

    label = classes[idx]
    conf = confidence.item()*100

    freshness = "Fresh" if "fresh" in label else "Rotten"
    fruit = label.replace("fresh","").replace("rotten","").capitalize()

    return fruit, conf, freshness

# --------------------------------------------------
# INPUT
# --------------------------------------------------
st.markdown("### Select Fruit")
fruit_choice = st.selectbox("",["Apple","Banana","Orange"])

st.markdown("### Capture Image")
camera_image = st.camera_input("Take a picture")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if camera_image is not None:
    image = Image.open(camera_image)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Analyze Freshness", use_container_width=True):
        fruit, conf, freshness = predict(image)

        st.markdown("## Result")

        if freshness=="Fresh":
            st.markdown(f"<div class='result-good'>✅ {fruit} is FRESH</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-bad'>❌ {fruit} is ROTTEN</div>", unsafe_allow_html=True)

        st.progress(min(int(conf),100))
        st.write(f"Confidence: **{conf:.2f}%**")


