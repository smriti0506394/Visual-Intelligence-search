
from pathlib import Path

import numpy as np
import open_clip
import streamlit as st
import torch
from PIL import Image

st.set_page_config(
    page_title="Visual Intelligence Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Visual Intelligence Search")
st.write("Find visually similar objects using AI.")

EMBEDDINGS_FILE = Path("artifacts/image_embeddings.npz")

@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    return model, preprocess, open_clip.get_tokenizer("ViT-B-32")

@st.cache_data
def load_embeddings():
    data = np.load(EMBEDDINGS_FILE)
    return data["embeddings"], data["paths"]

if not EMBEDDINGS_FILE.exists():
    st.error("Embeddings file not found. Run create_embeddings.py first.")
    st.stop()

try:
    embeddings, image_paths = load_embeddings()
    model, preprocess, tokenizer = load_model()
except Exception as exc:
    st.error(f"Could not load the search model: {exc}")
    st.stop()

query = st.text_input(
    "What are you looking for?",
    placeholder="Try: tv, person, car, dog..."
)
top_k = st.slider("Number of results", 1, min(10, len(image_paths)), 5)


def show_results(scores, title):
    indices = np.argsort(scores)[::-1][:top_k]
    st.subheader(title)

    columns = st.columns(min(3, len(indices)))

    for rank, idx in enumerate(indices):
        path = Path(str(image_paths[idx]))

        with columns[rank % len(columns)]:
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.write(f"**Rank:** {rank + 1}")
                st.write(f"**Similarity:** {scores[idx]:.4f}")
                st.caption(path.name)
            else:
                st.warning(f"Image not found: {path}")


# Text-based search
if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching images..."):
            tokens = tokenizer([query.strip()])

            with torch.no_grad():
                text_features = model.encode_text(tokens)
                text_features /= text_features.norm(
                    dim=-1, keepdim=True
                )

            scores = (
                embeddings @ text_features.cpu().numpy().T
            ).flatten()

        show_results(scores, f"Top results for: {query}")


# Webcam image search
st.divider()
st.subheader("📷 Search using your camera")

camera_photo = st.camera_input("Take a photo")

if camera_photo is not None:
    image = Image.open(camera_photo).convert("RGB")
    st.image(image, caption="Captured photo", width=350)

    if st.button("Find similar images", type="primary"):
        with st.spinner("Finding similar images with CLIP..."):
            image_tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(
                    dim=-1, keepdim=True
                )

            scores = (
                embeddings @ image_features.cpu().numpy().T
            ).flatten()

        show_results(scores, "Top matches for your photo")
