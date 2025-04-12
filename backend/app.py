import os
import uuid
import subprocess
import requests
from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
import json
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "static/uploads"
ALIGNED_FOLDER = "static/aligned"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ALIGNED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

HF_TOKEN = os.environ.get("GESTALT_HF_TOKEN", None)
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

HF_REPO = "kevins4202/GestaltMML"
MODEL_FILENAME = "GestaltMML_model.pt"
CROPPER_FILENAME = "Resnet50_Final.pth"

MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)
CROPPER_PATH = os.path.join(MODEL_FOLDER, CROPPER_FILENAME)

# 1. Download model weights from Hugging Face if not already present
def download_from_huggingface(filename, save_path):
    if os.path.exists(save_path):
        print(f"[INFO] {filename} already exists. Skipping download.")
        return
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
    print(f"[INFO] Downloading {filename} from {url}")
    response = requests.get(url, headers=HF_HEADERS)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"[INFO] Saved {filename} to {save_path}")
    else:
        raise Exception(f"Failed to download {filename}: {response.status_code}, {response.text}")

download_from_huggingface(MODEL_FILENAME, MODEL_PATH)
download_from_huggingface(CROPPER_FILENAME, CROPPER_PATH)

# 2. Load processor and model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

label_ids = list(range(528))
id2label = {i: str(i) for i in label_ids}
label2id = {str(i): i for i in label_ids}

model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-mlm", id2label=id2label, label2id=label2id
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 3. Load disease dictionary
with open("disease_dict.json") as f:
    disease_dict = json.load(f)

# 4. Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    question = request.form.get("question", "What disease does the patient have?")

    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    img_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.jpg")
    file.save(input_path)

    # Run crop_align.py with Resnet50_Final.pth
    output_path = os.path.join(ALIGNED_FOLDER, f"{img_id}_aligned.jpg")
    subprocess.run([
        "python3", "crop_align.py",
        "--data", input_path,
        "--save_dir", output_path,
        "--model_path", CROPPER_PATH,
        "--no_cuda"
    ], check=True)

    if not os.path.exists(output_path):
        return jsonify({"error": "Face alignment failed"}), 500

    # Run inference
    image = Image.open(output_path)
    inputs = processor(image, question, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.topk(logits, k=3).indices[0].tolist()

    predicted_diseases = [disease_dict[str(i)] for i in predicted]

    return jsonify({
        "prediction": predicted_diseases,
        "confidence": [round(100 * torch.softmax(logits, dim=1)[0][i].item(), 2) for i in predicted]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
