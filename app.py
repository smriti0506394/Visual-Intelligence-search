
from pathlib import Path

import numpy as np
import open_clip
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import threading

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



class CameraProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.frame = image.copy()

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


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
st.subheader("📷 Camera Search")

if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

if "camera_image" not in st.session_state:
    st.session_state.camera_image = None

# Open / Close camera
if st.button(
    "📷 Open Camera" if not st.session_state.show_camera
    else "Close Camera"
):
    st.session_state.show_camera = not st.session_state.show_camera
    st.session_state.camera_image = None
    st.rerun()

if st.session_state.show_camera:

    ctx = webrtc_streamer(
    key="visual-intelligence-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=CameraProcessor,
    media_stream_constraints={
        "video": {"facingMode": "user"},
        "audio": False
    },
    video_html_attrs={
        "autoPlay": True,
        "playsInline": True,
        "muted": True,
        "controls": False,
        "style": {
            "width": "100%",
            "height": "auto",
            "transform": "scaleX(-1)"
        }
    }
)

    # Capture photo
    if st.button("📸 Capture Photo"):

        if ctx.video_processor is not None:

            with ctx.video_processor.lock:
                frame = (
                    ctx.video_processor.frame.copy()
                    if ctx.video_processor.frame is not None
                    else None
                )

            if frame is not None:
                rgb_image = frame[:, :, ::-1].copy()
                st.session_state.camera_image = Image.fromarray(
                    rgb_image
                )
            else:
                st.warning("Camera frame is not ready yet.")

        else:
            st.warning("Please start the camera first.")

    # Show captured photo
    if st.session_state.camera_image is not None:

        st.image(
            st.session_state.camera_image,
            caption="Captured Photo",
            width=350
        )

        # CLIP image search
        if st.button("🔍 Find Similar Images"):

            with st.spinner("Finding similar images with CLIP..."):

                image_tensor = preprocess(
                    st.session_state.camera_image
                ).unsqueeze(0)

                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)

                    image_features /= image_features.norm(
                        dim=-1,
                        keepdim=True
                    )

                scores = (
                    embeddings @ image_features.cpu().numpy().T
                ).flatten()

            show_results(
                scores,
                "Top matches for your camera photo"
            )
