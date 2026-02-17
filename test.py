import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

# 1. Define the classes
classes = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

# 2. Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Recreate the Model Architecture
# We do NOT change the fc layer this time, because the training script didn't change it.
model = models.resnet18(pretrained=False)

# 4. Load the saved state dictionary
model_path = 'resnet18fruit_V001.pth'  # Make sure this matches your file name
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find model file '{model_path}'")
    exit()
except RuntimeError as e:
    print(f"Model loading error: {e}")
    exit()

# Move to device and evaluate
model = model.to(device)
model.eval()

# 5. Preprocessing (Must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # The model outputs 1000 scores, but we only care about the highest one
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)
            
            # Safety check: If the model predicts an index > 5 (outside our fruits), handle it
            idx = predicted_idx.item()
            if idx >= len(classes):
                print(f"Warning: Model predicted index {idx}, which is outside your class list.")
                return "Unknown", 0.0
            
            predicted_class = classes[idx]
            confidence = probs[0][idx].item() * 100
            
            return predicted_class, confidence

    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    # REPLACE with your actual image path
    image_to_test = "C:/Users/Suresh/Downloads/Screen Shot 2018-06-12 at 8.47.41 PM.png" 
    
    prediction, confidence = predict_image(image_to_test)
    
    if prediction:
        print(f"Prediction: {prediction} ({confidence:.2f}%)")