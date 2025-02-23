import torch
from PIL import Image
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_config():
    import yaml
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_vqgan_model():
    config = load_config()
    vqgan_config = config.get("vqgan", {})
    # Pseudo-code: Replace with actual VQGAN loading routine
    # For example, using a function from taming-transformers:
    from taming.models.vqgan import VQModel
    model = VQModel.load_from_config(vqgan_config.get("config_path"), vqgan_config.get("checkpoint_path"))
    model.to(device)
    model.eval()
    # Add latent dimension attribute if not present
    if not hasattr(model, "latent_dim"):
        model.latent_dim = vqgan_config.get("latent_dim", 256)
    return model

def decode_latent(model, latent):
    # Pseudo-code: Replace with your model's decode function
    # For instance, VQGAN might require additional processing to convert the latent to an image.
    image_tensor = model.decode(latent)
    # Convert tensor to a PIL image
    image_array = (image_tensor.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(image_array)
