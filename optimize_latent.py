import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from models.clip_utils import load_clip_model, get_image_embedding
from models.vqgan import load_vqgan_model, decode_latent

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
        # Generate image from latent
        image = decode_latent(vqgan, latent)
        # Get image embedding
        image_features = get_image_embedding(image, clip_model, preprocess)
        similarity = compute_similarity(text_features, image_features)
        loss = -similarity  # maximize similarity
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iteration {i}: Loss {loss.item()}, Similarity {similarity}")
    
    return latent

def main():
    parser = argparse.ArgumentParser(description="Optimize latent vector for image generation.")
    parser.add_argument("--iterations", type=int, default=300, help="Number of optimization iterations.")
    args = parser.parse_args()
    
    clip_model, preprocess = load_clip_model()
    vqgan = load_vqgan_model()
    
    # Typically, you would pass a text prompt through GPT-3 and then get text features using CLIP.
    dummy_text = "A futuristic cityscape at dusk with neon lights and flying cars"
    from models.clip_utils import get_text_embedding
    text_features = get_text_embedding(dummy_text, clip_model)
    
    optimized_latent = optimize_latent(vqgan, clip_model, preprocess, text_features, iterations=args.iterations)
    final_image = decode_latent(vqgan, optimized_latent)
    final_image.save("assets/generated_image.png")
    print("Image saved to assets/generated_image.png")

if __name__ == "__main__":
    main()
