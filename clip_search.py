
from pathlib import Path
import json

import torch
import open_clip
from PIL import Image

# Project settings
IMAGE_DIR = Path("data/image")
OUTPUT_DIR = Path("outputs")
EMBEDDINGS_PATH = OUTPUT_DIR / "clip_embeddings.pt"

DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
TOP_K = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)
model = model.to(DEVICE).eval()

# Collect supported images
image_paths = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower()
    in {".jpg", ".jpeg", ".png", ".webp"}
)

if not image_paths:
    raise SystemExit(f"No images found in {IMAGE_DIR.resolve()}")

def encode_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(
            dim=-1, keepdim=True
        )
    return embedding.cpu().squeeze(0)

def build_index():
    embeddings = []
    valid_paths = []

    for i, path in enumerate(image_paths, start=1):
        try:
            embeddings.append(encode_image(path))
            valid_paths.append(str(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)} images")

    if not embeddings:
        raise SystemExit("Could not encode any images.")

    data = {
        "paths": valid_paths,
        "embeddings": torch.stack(embeddings),
    }
    torch.save(data, EMBEDDINGS_PATH)
    print(f"Indexed {len(valid_paths)} images.")
    print(f"Saved index to {EMBEDDINGS_PATH}")

def search(query_path):
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit(
            "Index not found. Run: python clip_search.py index"
        )

    data = torch.load(
        EMBEDDINGS_PATH,
        map_location="cpu",
        weights_only=True
    )
    query_path = Path(query_path)

    if not query_path.is_file():
        raise SystemExit(f"Query image not found: {query_path}")

    query_embedding = encode_image(query_path)
    matrix = data["embeddings"]

    # Embeddings are normalized, so dot product gives cosine similarity
    scores = matrix @ query_embedding

    k = min(TOP_K, len(data["paths"]))
    values, indices = torch.topk(scores, k=k)

    print(f"\nTop {k} matches for: {query_path}\n")
    for rank, (score, idx) in enumerate(
        zip(values.tolist(), indices.tolist()), start=1
    ):
        print(
            f"{rank}. {data['paths'][idx]} "
            f"(similarity: {score:.4f})"
        )

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python clip_search.py index")
        print("  python clip_search.py search path\\to\\query.jpg")
    elif sys.argv[1].lower() == "index":
        build_index()
    elif sys.argv[1].lower() == "search" and len(sys.argv) >= 3:
        search(sys.argv[2])
    else:
        print("Invalid command. Check the usage above.")
