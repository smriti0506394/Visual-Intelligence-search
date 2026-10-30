
from pathlib import Path
import numpy as np
import torch
import open_clip
from PIL import Image

# Folders
CROP_DIR = Path("outputs/crops")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Find cropped images
image_paths = sorted(
    p for p in CROP_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise SystemExit(
        "No cropped images found. Run main.py first."
    )

print(f"Found {len(image_paths)} cropped images.")

# Load a small pretrained CLIP model
device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(device)
model.eval()

embeddings = []
valid_paths = []

# Process small batches to save memory
batch_size = 8

with torch.no_grad():
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    batch_images.append(
                        preprocess(img.convert("RGB"))
                    )
                    valid_paths.append(str(path))
            except Exception as exc:
                print(f"Skipping {path.name}: {exc}")

        if not batch_images:
            continue

        image_tensor = torch.stack(batch_images).to(device)
        features = model.encode_image(image_tensor)
        features /= features.norm(dim=-1, keepdim=True)
        embeddings.append(features.cpu().numpy())

        print(
            f"Processed {min(start + batch_size, len(image_paths))}"
            f" / {len(image_paths)}"
        )

if not embeddings:
    raise SystemExit("No valid image embeddings were generated.")

all_embeddings = np.concatenate(embeddings, axis=0)

np.savez_compressed(
    ARTIFACTS_DIR / "image_embeddings.npz",
    embeddings=all_embeddings,
    paths=np.array(valid_paths)
)

print("Embedding generation complete!")
print(f"Embedding shape: {all_embeddings.shape}")
print("Saved to artifacts/image_embeddings.npz")
