
from pathlib import Path
import csv
from ultralytics import YOLO

# Project folders
IMAGE_DIR = Path("data/image")
OUTPUT_DIR = Path("outputs")
CROP_DIR = OUTPUT_DIR / "crops"
CSV_PATH = OUTPUT_DIR / "detections.csv"

CROP_DIR.mkdir(parents=True, exist_ok=True)

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Process only the first 10 images
image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)[:10]

with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "image_name", "object_name", "confidence", "crop_path"
    ])

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=640,
            device="cpu",
            verbose=False
        )

        result = results[0]

        if result.boxes is None:
            continue

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            # Keep crop coordinates inside image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(result.orig_shape[1], x2)
            y2 = min(result.orig_shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(box.cls[0].item())
            object_name = result.names[class_id]
            confidence = float(box.conf[0].item())

            crop = result.orig_img[y1:y2, x1:x2]
            crop_name = (
                f"{image_path.stem}_{i}_{object_name}.jpg"
            )
            crop_path = CROP_DIR / crop_name

            import cv2
            cv2.imwrite(str(crop_path), crop)

            writer.writerow([
                image_path.name,
                object_name,
                round(confidence, 4),
                str(crop_path)
            ])

print(f"Processed {len(image_paths)} images.")
print(f"Object crops saved in: {CROP_DIR}")
print(f"Detection details saved in: {CSV_PATH}")
