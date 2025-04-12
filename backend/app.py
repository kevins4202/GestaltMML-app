import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import io
from huggingface_hub import hf_hub_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
HF_MODEL_REPO = "your-username/your-model-name"  # Replace with your Hugging Face repo
HF_MODEL_FILENAME = "model.pt"  # The filename of your model in the repo
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)  # Set this in your environment variables for private repos

# Assuming your model outputs class indices, map them to disease names
DISEASE_CLASSES = {
    0: "Healthy",
    1: "Disease A",
    2: "Disease B",
    # Add more classes as per your model
}

def load_model_from_huggingface():
    """Download and load model from Hugging Face Hub"""
    global MODEL
    
    try:
        logger.info(f"Downloading model from Hugging Face: {HF_MODEL_REPO}")
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILENAME,
            token=HF_TOKEN
        )
        
        logger.info("Loading model into memory")
        MODEL = torch.load(model_path, map_location=DEVICE)
        MODEL.eval()  # Set to evaluation mode
        logger.info(f"Model loaded successfully on {DEVICE}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess the image for model input"""
    # Adjust these transformations based on your model's requirements
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image.to(DEVICE)

@app.before_request
def initialize():
    global MODEL
    if MODEL is None:
        success = load_model_from_huggingface()
        if not success:
            # If model loading fails, we'll handle requests individually

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the service is running and model is loaded"""
    if MODEL is None:
        return jsonify({'status': 'Service running, but model not loaded'}), 503
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    global MODEL
    
    # Check if model is loaded
    if MODEL is None:
        success = load_model_from_huggingface()
        if not success:
            return jsonify({'error': 'Failed to load model'}), 500
    
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess image
        img_tensor = preprocess_image(img)
        
        # Get prediction
        with torch.no_grad():
            output = MODEL(img_tensor)
            
        # Process the prediction (adjust based on your model output)
        _, predicted_idx = torch.max(output, 1)
        predicted_disease = DISEASE_CLASSES.get(predicted_idx.item(), "Unknown")
        
        # Get confidence scores
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence = probabilities[predicted_idx].item() * 100
        
        # Return all class probabilities for more detailed analysis
        all_probs = {DISEASE_CLASSES.get(i, f"Class {i}"): float(prob) * 100 
                    for i, prob in enumerate(probabilities.cpu().numpy())}
        
        return jsonify({
            'prediction': predicted_disease,
            'confidence': f"{confidence:.2f}%",
            'probabilities': all_probs
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to load the model on startup
    load_model_from_huggingface()
    
    # Start the Flask server
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)