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
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

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

# download_from_huggingface(MODEL_FILENAME, MODEL_PATH)
# download_from_huggingface(CROPPER_FILENAME, CROPPER_PATH)
gestalt_model_path = hf_hub_download(repo_id="kevins4202/GestaltMML", filename="GestaltMML_model.pt", token=HF_TOKEN)
resnet_path = hf_hub_download(repo_id="kevins4202/GestaltMML", filename="Resnet50_Final.pth", token=HF_TOKEN)

# 2. Load processor and model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

label_ids = list(range(528))
id2label = {i: str(i) for i in label_ids}
label2id = {str(i): i for i in label_ids}

model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-mlm", id2label=id2label, label2id=label2id
)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.load_state_dict(torch.load(gestalt_model_path, map_location=device))
model.to(device)
model.eval()

# 3. Load disease dictionary
with open("disease_dict.json") as f:
    disease_dict = json.load(f)

# 4. Prediction route
@app.route("/api/predict", methods=["POST"])
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
        "--cropper_model", resnet_path,
    ], check=True)

    if not os.path.exists(output_path) or not os.path.isfile(output_path + "/" + os.path.basename(output_path)):
        return jsonify({"error": "Face alignment failed"}), 500

    print(f"[INFO] Alignment successful: {output_path}")

    # Run inference
    image = Image.open(output_path + "/" + os.path.basename(output_path))
    inputs = processor(image, question, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.topk(logits, k=3).indices[0].tolist()

        probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities

        topk = torch.topk(probs, k=3)
        topk_probs = topk.values[0].tolist()
        topk_indices = topk.indices[0].tolist()

    predicted_diseases = [disease_dict[str(i)] for i in topk_indices]

    # Combine diseases with their confidences
    predictions_with_confidence = list(zip(predicted_diseases, topk_probs))

    print(f"[INFO] Inference successful: {predictions_with_confidence}")

    return jsonify({
        "predictions": predictions_with_confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
