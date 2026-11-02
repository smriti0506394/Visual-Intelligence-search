import cv2
from pathlib import Path
import threading

import av
import numpy as np
import open_clip
import streamlit as st
import torch
import cv2

from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Visual Intelligence Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Visual Intelligence Search")
st.write("Find visually similar objects using AI.")


# ---------------- FILE PATHS ----------------

EMBEDDINGS_FILE = Path("artifacts/image_embeddings.npz")
YOLO_MODEL_FILE = Path("yolo11n.pt")


# ---------------- CLIP MODEL ----------------

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


# ---------------- YOLO MODEL ----------------

@st.cache_resource
def load_yolo_model():
    return YOLO(str(YOLO_MODEL_FILE))


# ---------------- LOAD MODELS AND DATA ----------------

if not EMBEDDINGS_FILE.exists():
    st.error(
        "Embeddings file not found. Run create_embeddings.py first."
    )
    st.stop()

try:
    embeddings, image_paths = load_embeddings()

    model, preprocess, tokenizer = load_model()

except Exception as exc:
    st.error(f"Could not load the CLIP search model: {exc}")
    st.stop()




# ---------------- DISPLAY CLIP SEARCH RESULTS ----------------

def show_results(scores, title):

    indices = np.argsort(scores)[::-1][:top_k]

    st.subheader(title)

    columns = st.columns(min(3, len(indices)))

    for rank, idx in enumerate(indices):

        path = Path(str(image_paths[idx]))

        with columns[rank % len(columns)]:

            if path.exists():

                st.image(
                    str(path),
                    use_container_width=True
                )

                st.write(f"**Rank:** {rank + 1}")

                st.write(
                    f"**Similarity:** {scores[idx]:.4f}"
                )

                st.caption(path.name)

            else:

                st.warning(f"Image not found: {path}")


# ---------------- YOLO DETECTION FUNCTION ----------------

def detect_objects(image, title, confidence):

    if not YOLO_MODEL_FILE.exists():

        st.error("YOLO model file yolo11n.pt was not found.")

        return

    try:

        with st.spinner("YOLO is detecting objects..."):

            yolo_model = load_yolo_model()

            results = yolo_model.predict(
                source=image,
                imgsz=640,
                conf=confidence,
                device="cpu",
                verbose=False
            )

            result = results[0]

        st.subheader(title)

        if result.boxes is None or len(result.boxes) == 0:

            st.warning(
                "No objects detected. Try lowering the confidence threshold."
            )

            return

        st.success(
            f"Detected {len(result.boxes)} object(s)."
        )

        # Display annotated image
        annotated_image = result.plot()

        st.image(
            annotated_image[:, :, ::-1],
            caption="YOLO Object Detection",
            use_container_width=True
        )

        # Display individual object crops
        st.subheader("Detected Object Crops")

        crop_columns = st.columns(3)

        original_image = result.orig_img

        image_height, image_width = original_image.shape[:2]

        for i, box in enumerate(result.boxes):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Keep coordinates inside image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(image_width, x2)
            y2 = min(image_height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(box.cls[0].item())

            object_name = result.names[class_id]

            confidence_score = float(box.conf[0].item())

            crop = original_image[y1:y2, x1:x2]

            crop_rgb = crop[:, :, ::-1]

            with crop_columns[i % 3]:

                st.image(
                    crop_rgb,
                    caption=(
                        f"{object_name} "
                        f"({confidence_score:.2%})"
                    ),
                    use_container_width=True
                )

    except Exception as exc:

        st.error(f"YOLO detection failed: {exc}")



# ---------------- TEXT-BASED CLIP SEARCH ----------------

st.divider()

st.header("🔍 Search using text")

# Search input comes directly below the heading
query = st.text_input(
    "What are you looking for?",
    placeholder="Try: tv, person, car, dog...",
    key="text_search_query"
)

# Number of results
top_k = st.slider(
    "Number of results",
    1,
    min(10, len(image_paths)),
    min(5, len(image_paths))
)

# Search button comes after the input
if st.button("🔍 Search", type="primary", key="text_search"):

    if not query.strip():

        st.warning("Please enter a search query.")

    else:

        with st.spinner("Searching images..."):

            tokens = tokenizer([query.strip()])

            with torch.no_grad():

                text_features = model.encode_text(tokens)

                text_features /= text_features.norm(
                    dim=-1,
                    keepdim=True
                )

            scores = (
                embeddings @ text_features.cpu().numpy().T
            ).flatten()

        show_results(
            scores,
            f"Top results for: {query}"
        )



# ---------------- UPLOADED IMAGE YOLO DETECTION ----------------

st.divider()

st.header("🎯 Object Detection using Uploaded Images")

uploaded_image = st.file_uploader(
    "Upload an image for YOLO detection",
    type=["jpg", "jpeg", "png"],
    key="yolo_image_upload"
)

confidence_threshold = st.slider(
    "YOLO confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
    key="yolo_confidence"
)

if uploaded_image is not None:

    upload_image = Image.open(uploaded_image).convert("RGB")

    st.image(
        upload_image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "Detect Objects with YOLO",
        key="uploaded_yolo"
    ):

        detect_objects(
            upload_image,
            "Uploaded Image Detection Results",
            confidence_threshold
        )


# ---------------- LIVE CAMERA PROCESSOR ----------------


class CameraProcessor(VideoProcessorBase):

    def __init__(self):

        self.frame = None
        self.lock = threading.Lock()

        self.yolo_model = YOLO("yolo11n.pt")
        self.confidence = 0.25

        self.frame_count = 0
        self.detections = []

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # Check actual camera resolution
        print("Camera resolution:", image.shape)

        # Correct camera orientation
        image = np.fliplr(image).copy()

        with self.lock:
            self.frame = image.copy()

        try:

            self.frame_count += 1

            # Run YOLO every 4 frames
            if self.frame_count % 4 == 0:

                results = self.yolo_model.predict(
                    source=image,
                    imgsz=416,
                    conf=self.confidence,
                    device="cpu",
                    verbose=False
                )

                result = results[0]

                self.detections = []

                if result.boxes is not None:

                    for box in result.boxes:

                        x1, y1, x2, y2 = map(
                            int, box.xyxy[0].tolist()
                        )

                        class_id = int(box.cls[0].item())
                        name = result.names[class_id]
                        score = float(box.conf[0].item())

                        self.detections.append(
                            (x1, y1, x2, y2, name, score)
                        )

            # Draw latest detections on current frame
            output = image.copy()

            for x1, y1, x2, y2, name, score in self.detections:

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{name} {score:.0%}"

                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        except Exception as exc:

            print("Detection error:", exc)
            output = image

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )



# ---------------- WEBCAM CAMERA SEARCH ----------------

st.divider()

st.header("📷 Camera Search")

if "show_camera" not in st.session_state:

    st.session_state.show_camera = False

if "camera_image" not in st.session_state:

    st.session_state.camera_image = None


# ---------------- OPEN / CLOSE CAMERA ----------------

if st.button(
    "📷 Open Camera" if not st.session_state.show_camera
    else "Close Camera",
    key="camera_toggle"
):

    st.session_state.show_camera = not st.session_state.show_camera
    st.session_state.camera_image = None
    st.session_state.camera_started = False

    st.rerun()


# ---------------- WEBCAM STREAM ----------------

if st.session_state.show_camera:

    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=CameraProcessor,

        media_stream_constraints={
    "video": {
        "width": {"exact": 1280},
        "height": {"exact": 720},
        "frameRate": {"ideal": 30}
    },
    "audio": False
},

        async_processing=True,

        video_html_attrs={
            "autoPlay": True,
            "playsInline": True,
            "muted": True,
            "controls": False,
            "style": {
                "width": "100%",
                "height": "auto",
                "transform": "scaleX(1)"
            }
        }
    )

    st.session_state.camera_started = ctx.state.playing
  # ---------------- CAPTURE CAMERA PHOTO ----------------

if st.button(
    "📸 Capture Photo",
    key="capture_photo"
):

    if st.session_state.camera_started:

        if ctx.video_processor is not None:

            with ctx.video_processor.lock:

                frame = (
                    ctx.video_processor.frame.copy()
                    if ctx.video_processor.frame is not None
                    else None
                )

            if frame is not None:

                rgb_image = frame[:, :, ::-1].copy()

                # Match captured photo with mirrored webcam preview
                rgb_image = np.fliplr(rgb_image).copy()

                st.session_state.camera_image = Image.fromarray(
                    rgb_image
                )

                st.success("Photo captured successfully!")

            else:
                st.warning("Camera frame is not ready yet.")

        else:
            st.warning("Camera is not initialized.")

    else:
        st.warning("Please start the camera first.")


# ---------------- DISPLAY CAPTURED PHOTO ----------------

if st.session_state.camera_image is not None:

    camera_image = st.session_state.camera_image

    st.divider()

    st.subheader("Captured Photo")

    st.image(
        camera_image,
        caption="Your captured camera photo",
        width=350
    )
# ---------------- CLIP CAMERA SEARCH ----------------

if st.session_state.camera_image is not None:

    camera_image = st.session_state.camera_image

    st.divider()
    st.subheader("🔎 Analyze Captured Photo")

    # ---------------- CLIP CAMERA SEARCH ----------------

    if st.button(
        "🔍 Find Similar Images with CLIP",
        key="camera_clip_search"
    ):

        with st.spinner(
            "Finding similar images with CLIP..."
        ):

            image_tensor = preprocess(
                camera_image
            ).unsqueeze(0)

            with torch.no_grad():

                image_features = model.encode_image(
                    image_tensor
                )

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

    # ---------------- YOLO CAMERA PHOTO DETECTION ----------------

    if st.button(
        "🎯 Detect Objects with YOLO",
        key="camera_yolo_detection"
    ):

        detect_objects(
            camera_image,
            "YOLO Camera Detection Results",
            confidence_threshold
        )

else:

    st.info("📸 Capture a photo first to use CLIP or YOLO.")