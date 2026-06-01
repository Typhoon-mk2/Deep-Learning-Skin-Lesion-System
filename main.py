from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# label

class_names = [
    "bkl",
    "nv",
    "df",
    "mel",
    "vasc",
    "bcc",
    "akiec"
]


# loading model

model = models.resnet18(pretrained=False)

model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 7)
)

model.load_state_dict(
    torch.load("./model/ResNet18_model.pth", map_location=device)
)
model.to(device)
model.eval()


# picture transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# routes


@app.route("/", methods=["GET", "POST"])
def index():
    disease_map = {
        "bkl" : 'Benign keratosis-like lesions',
        "nv" : 'melanocytic nevi',
        "df" : 'dermatofibroma',
        "mel": 'melanoma',
        "vasc": 'vascular lesions',
        "bcc": 'Basal cell carcinoma',
        "akiec" : 'Actinic keratoses and intraepithelial carcinoma'
    }

    prediction = None
    confidence = 0

    if request.method == "POST":

        file = request.files["file"]

        if file:

            path = os.path.join("static/uploads", file.filename)
            file.save(path)

            img = Image.open(path).convert("RGB")

            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():

                output = model(img)

                probs = torch.softmax(output, dim=1)

                pred = torch.argmax(probs, dim=1).item()

                confidence = probs[0][pred].item()

            disease_name = disease_map[class_names[pred]]

            prediction = disease_name

    return render_template(
        "index.html",
        prediction=prediction,
        confidence = round(confidence*100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)