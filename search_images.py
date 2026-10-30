
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

# Project paths
CROP_DIR = Path("outputs/crops")
EMBEDDINGS_FILE = Path("artifacts/image_embeddings.npz")

# Load saved image embeddings and paths
data = np.load(EMBEDDINGS_FILE)
embeddings = data["embeddings"]
image_paths = data["paths"]

# Load the same CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to("cpu")
model.eval()

def search_by_text(query, top_k=5):
    # Convert text into a CLIP embedding
    tokens = tokenizer([query])

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(
            dim=-1, keepdim=True
        )

    # Compare text with every image embedding
    similarities = embeddings @ text_features.cpu().numpy().T
    similarities = similarities.flatten()

    # Get the most similar images
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\nSearch query: {query}")
    print("-" * 50)

    for rank, idx in enumerate(top_indices, start=1):
        print(f"{rank}. {image_paths[idx]}")
        print(f"   Similarity: {similarities[idx]:.4f}")

if __name__ == "__main__":
    query = input("Enter an object to search for: ")
    search_by_text(query)
