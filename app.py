import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
from model import  DualImageRegressor  # importa la tua classe modello

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model
model = DualImageRegressor()
model.load_state_dict(torch.load("modello_5.pt", map_location=DEVICE))
model.eval()
model.to(DEVICE)

# Parametri normalizzazione (gli stessi usati nel training)
mean_year = 128.40
std_year = 16.80

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'front' not in request.files or 'back' not in request.files:
        return jsonify({"error": "Invia entrambe le immagini: front e back"}), 400

    try:
        front_img = Image.open(request.files['front']).convert("RGB")
        back_img = Image.open(request.files['back']).convert("RGB")
        front_tensor = transform(front_img).unsqueeze(0).to(DEVICE)
        back_tensor = transform(back_img).unsqueeze(0).to(DEVICE)

##torch.no_grad() è un context manager di PyTorch che disattiva il calcolo automatico del gradiente.
        with torch.no_grad():
            output = model(front_tensor, back_tensor)
            prediction = output.item() * std_year + mean_year

        return jsonify({"predicted_year": int(round(prediction, 2))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
