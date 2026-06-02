
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes_binary = ["Real", "Fake"]

print("Device:", device)

from torchvision import transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform_deepfake = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    normalize
])

transform_morph = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_splice = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# from tensorflow.keras.models import load_model

import torchvision.models as models
import torch.nn as nn
import torch

def load_morph_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_resnet_dropout_model(path):
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

model1 = load_resnet_dropout_model("deepfake_resnet18_v2.pth")  # Deepfake
model2 = load_morph_model("morph_modell.pth")        # Morph
model3 = load_resnet_dropout_model("resnet18_splicing.pth")  # Splice

print("✅ ALL MODELS LOADED PERFECTLY")

import torch
import torch.nn.functional as F

def ensemble_decision(img_pil):

    model1.eval()
    model2.eval()
    model3.eval()

    img1 = transform_deepfake(img_pil).unsqueeze(0).to(device)
    img2 = transform_morph(img_pil).unsqueeze(0).to(device)
    img3 = transform_splice(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # Deepfake
        out1 = F.softmax(model1(img1), dim=1)

        #  Morph
        morph_prob = torch.sigmoid(model2(img2))   # [1,1]
        out2 = torch.cat([1 - morph_prob, morph_prob], dim=1)  # [1,2]

        # Splicing
        out3 = F.softmax(model3(img3), dim=1)

    # (agar training reversed thi tabhi rakho)
    out1 = out1[:, [1, 0]]

    # Ensemble
    avg_probs = (out1 + out2 + out3) / 3
    final_pred = torch.argmax(avg_probs, 1).item()
    confidence = avg_probs[0][final_pred].item() * 100

    p1 = torch.argmax(out1, 1).item()
    p2 = torch.argmax(out2, 1).item()
    p3 = torch.argmax(out3, 1).item()

    print("\n🔍 Individual Models:")
    print(f"Deepfake Model → {classes_binary[p1]} ({out1[0][p1].item()*100:.2f}%)")
    print(f"Morph Model → {classes_binary[p2]} ({out2[0][p2].item()*100:.2f}%)")
    print(f"Splicing Model → {classes_binary[p3]} ({out3[0][p3].item()*100:.2f}%)")

    print("\n🎯 Final Ensemble Decision (Soft Voting):")
    print(f"{classes_binary[final_pred]} ({confidence:.2f}%)")

    return final_pred, confidence

import torch.nn.functional as F


def detect_fake_type(imgs):

    img1, img2, img3 = imgs

    model1.eval()
    model2.eval()
    model3.eval()

    with torch.no_grad():
        deepfake_out = F.softmax(model1(img1), dim=1)

        morph_out = torch.sigmoid(model2(img2))
        morph_out = torch.cat([1 - morph_out, morph_out], dim=1)

        splice_out = F.softmax(model3(img3), dim=1)

    deepfake_out = deepfake_out[:, [1, 0]]

    d_score = deepfake_out[0][1].item()
    m_score = morph_out[0][1].item()
    s_score = splice_out[0][1].item()

    if d_score < 0.5 and m_score < 0.5 and s_score < 0.5:
        return "Real"

    scores = {
        "Deepfake": d_score,
        "Morphed": m_score,
        "Cut-Paste": s_score
    }

    return max(scores, key=scores.get)

import os
import google.generativeai as genai
from PIL import Image

API_KEY = "KEYHERE"

genai.configure(api_key=API_KEY)

def check_gemini_synthid(img):
    """
    Passes the image to the free Gemini API with the specific @SynthID prompt.
    """
    try:
        # gemini-flash-latest is a robust choice for fast text-based models, similar to the working chatbot.
        model = genai.GenerativeModel('gemini-flash-latest')

        # The prompt exactly as you requested
        prompt = "@SynthID check this image. Reply ONLY with 'yes' if SynthId Detected, or 'no' if it is not."

        # Send both the text prompt and the PIL Image object to Gemini
        response = model.generate_content([prompt, img])

        # Parse the response
        answer = response.text.strip().lower()

        if 'yes' in answer:
            # If Gemini confirms it's AI, we return True
            return True, 99.9
        else:
            return False, 0.0

    except Exception as e:
        print(f"☢ Gemini API check failed or timed out: {e}")
        # If the API fails (e.g., no internet), default to False so the local ensemble takes over
        return False, 0.0

def final_pipeline(img_path):

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

    print("\n📸 Processing Image...\n")

    # --- Step 0: Gemini API @SynthID Check ---
    print("🔍 Asking Gemini to check for SynthID...")
    is_synthid_fake, synthid_conf = check_gemini_synthid(img)

    if is_synthid_fake:
        print(f"\n🚨 FINAL RESULT: FAKE IMAGE (AI Generated - Gemini Detection) ({synthid_conf:.2f}%)")
        return {
            "result": "Fake",
            "type": "AI Generated (SynthID/Gemini)",
            "confidence": synthid_conf
        }

    print("✅ Gemini returned 'No'. Proceeding to deepfake ensemble...\n")
    # -----------------------------------------

    # 🔥 Preprocess ONCE (Assuming your transform functions are defined elsewhere)
    img1 = transform_deepfake(img).unsqueeze(0).to(device)
    img2 = transform_morph(img).unsqueeze(0).to(device)
    img3 = transform_splice(img).unsqueeze(0).to(device)

    # Step 1: Ensemble
    decision, confidence = ensemble_decision(img)

    label = classes_binary[decision]

    # Step 2
    if label.lower() == "real":
        print(f"\n✅ FINAL RESULT: REAL IMAGE ({confidence:.2f}%)")
        return {
            "result": "Real",
            "confidence": confidence
        }

    else:
        fake_type = detect_fake_type((img1, img2, img3))
        print(f"\n🚨 FINAL RESULT: FAKE IMAGE ({fake_type}) ({confidence:.2f}%)")
        return {
            "result": "Fake",
            "type": fake_type,
            "confidence": confidence
        }

# final_pipeline("/content/Gemini_Generated_Image_dcmnejdcmnejdcmn.png")

