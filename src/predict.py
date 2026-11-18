"""Single-image prediction helper."""
import torch
from PIL import Image
from torchvision import transforms

from model import build_model


def load_predictor(checkpoint_path="outputs/best_model.pth"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    size = int(checkpoint["image_size"])
    classes = checkpoint["classes"]
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, transform


def predict_image(image: Image.Image, predictor):
    model, classes, transform = predictor
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    return classes[index], float(probabilities[index])
