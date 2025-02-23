
import argparse
from models.gpt3 import generate_description
from models.clip_utils import load_clip_model, get_text_embedding, get_image_embedding
from models.vqgan import load_vqgan_model, decode_latent
import torch
import torch.nn.functional as F
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_similarity(text_features, image_features):
    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)
    return (text_features @ image_features.T).item()

def optimize_latent(vqgan, clip_model, preprocess, text_features, iterations=300, lr=0.05):
    latent = torch.randn((1, vqgan.latent_dim), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([latent], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        image = decode_latent(vqgan, latent)
        image_features = get_image_embedding(image, clip_model, preprocess)
        similarity = compute_similarity(text_features, image_features)
        loss = -similarity
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iteration {i}: Loss {loss.item()}, Similarity {similarity}")
    
    return latent

def main():
    parser = argparse.ArgumentParser(description="Run the full text-to-image pipeline.")
    parser.add_argument("--prompt", type=str, required=True, help="Input text prompt.")
    args = parser.parse_args()
    
    # Generate detailed caption using GPT-3
    caption = generate_description(args.prompt)
    print("Generated Caption:", caption)
    
    # Load models
    clip_model, preprocess = load_clip_model()
    vqgan = load_vqgan_model()
    
    # Get text embedding
    text_features = get_text_embedding(caption, clip_model)
    
    # Optimize latent
    optimized_latent = optimize_latent(vqgan, clip_model, preprocess, text_features)
    
    # Generate final image
    final_image = decode_latent(vqgan, optimized_latent)
    final_image.save("assets/generated_image.png")
    print("Image saved to assets/generated_image.png")

if __name__ == "__main__":
    main()
