from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import os

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)  

model.load_state_dict(
    torch.load("fine_tuned_resnet50_animal_classifier.pth", map_location=torch.device('cpu'))
)
model.eval()


classes = ['cat', 'dog', 'wild']  


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = Image.open(filepath).convert('RGB')
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                prediction = classes[predicted.item()]

            image_path = filepath

    return render_template(
        'index.html',
        prediction=prediction,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)