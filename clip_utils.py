import torch
import clip
from PIL import Image
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    config = load_config()
    clip_config = config.get("clip", {})
    model, preprocess = clip.load(clip_config.get("model_name", "ViT-B/32"), device=device)
    return model, preprocess

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_text_embedding(text, model, device=device):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    return text_features

def get_image_embedding(image, model, preprocess, device=device):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features
