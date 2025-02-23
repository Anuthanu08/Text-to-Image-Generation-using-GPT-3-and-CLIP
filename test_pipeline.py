
import pytest
from models.gpt3 import generate_description
from models.clip_utils import load_clip_model, get_text_embedding
from models.vqgan import load_vqgan_model, decode_latent
import torch

def test_generate_description():
    prompt = "Test prompt"
    caption = generate_description(prompt)
    assert isinstance(caption, str) and len(caption) > 0

def test_clip_text_embedding():
    clip_model, _ = load_clip_model()
    text = "A sample description"
    text_features = get_text_embedding(text, clip_model)
    assert text_features is not None
    assert text_features.shape[0] == 1

def test_vqgan_decode():
    vqgan = load_vqgan_model()
    # Create a dummy latent vector with the correct dimensions
    latent = torch.randn((1, vqgan.latent_dim), device="cuda" if torch.cuda.is_available() else "cpu")
    image = decode_latent(vqgan, latent)
    # Check that an image is returned (PIL Image instance)
    from PIL import Image
    assert isinstance(image, Image.Image)
