import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------
# 1. Model setup
# -------------------------
class_names = ["normal", "reversal", "corrected"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

# Load weights with checks
state_dict = torch.load("best_mobilenetv2_dyslexia.pth", map_location=DEVICE)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys or unexpected_keys:
    st.warning(f"⚠️ Weight mismatch detected!\nMissing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")

model = model.to(DEVICE)
model.eval()

# -------------------------
# 2. Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_dyslexia(img):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probs.argmax().item()
        confidence = probs[predicted_idx].item()

    return class_names[predicted_idx], confidence, probs

# -------------------------
# 3. Streamlit UI
# -------------------------
st.title("📝 Dyslexia Detection from Handwriting")
st.write("Upload a handwriting image to check for dyslexia patterns.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        label, conf, probs = predict_dyslexia(img)

        # Store confidence in session_state
        st.session_state["last_confidence"] = conf
        st.session_state["last_label"] = label

        st.subheader(f"Prediction: **{label}** ({conf:.2%} confidence)")
        st.write("### Class probabilities:")
        for i, cls in enumerate(class_names):
            st.write(f"- **{cls}**: {probs[i].item():.2%}")

        # Warning if the model predicts 100% for one class (common sign of bad weights)
        if conf > 0.99:
            st.warning("⚠️ Model predicts one class with extremely high confidence. Check if the loaded weights match the architecture.")
