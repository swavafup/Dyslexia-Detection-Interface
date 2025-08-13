import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

# -------------------------
# 1. Configuration
# -------------------------
class_names = ["normal", "reversal", "corrected"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 2. Model setup
# -------------------------
# Create MobileNetV2 architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

# Load weights safely
state_dict = torch.load("best_mobilenetv2_dyslexia.pth", map_location=DEVICE)

# Strip 'module.' prefix if present (from DataParallel training)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
if missing_keys or unexpected_keys:
    st.warning(f"⚠️ Weight mismatch detected!\nMissing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")

model = model.to(DEVICE)
model.eval()

# -------------------------
# 3. Image preprocessing
# -------------------------
# Try to detect if the model expects 1 channel or 3 channels
first_conv = model.features[0][0]  # First conv layer
in_channels = first_conv.in_channels

if in_channels == 1:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# -------------------------
# 4. Prediction function
# -------------------------
def predict_dyslexia(img):
    if in_channels == 1:
        img = img.convert("L")  # grayscale
    else:
        img = img.convert("RGB")
        
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probs.argmax().item()
        confidence = probs[predicted_idx].item()

    return class_names[predicted_idx], confidence, probs

# -------------------------
# 5. Streamlit UI
# -------------------------
st.title("📝 Dyslexia Detection from Handwriting")
st.write("Upload a handwriting image to check for dyslexia patterns.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        label, conf, probs = predict_dyslexia(img)

        # Store in session_state
        st.session_state["last_confidence"] = conf
        st.session_state["last_label"] = label

        st.subheader(f"Prediction: **{label}** ({conf:.2%} confidence)")
        st.write("### Class probabilities:")
        for i, cls in enumerate(class_names):
            st.write(f"- **{cls}**: {probs[i].item():.2%}")

        # Warning for extremely high confidence
        if conf > 0.99:
            st.warning("⚠️ Model predicts one class with extremely high confidence. Verify that weights match the architecture.")
