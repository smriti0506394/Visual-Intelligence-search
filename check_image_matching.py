
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "image"
CSV_FILE = ROOT / "data" / "annotations" / "validation-images-with-rotation.csv"

print("Checking local images and annotation CSV...")

if not IMAGE_DIR.exists():
    print("ERROR: Image folder not found:", IMAGE_DIR)
    raise SystemExit(1)

if not CSV_FILE.exists():
    print("ERROR: CSV file not found:", CSV_FILE)
    raise SystemExit(1)

local_ids = {
    p.stem
    for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

open_images_ids = set()

with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        image_id = row.get("ImageID")
        if image_id:
            open_images_ids.add(image_id)

matching_ids = local_ids & open_images_ids

print("\n" + "=" * 40)
print("LOCAL IMAGE MATCHING REPORT")
print("=" * 40)
print("Local images:", len(local_ids))
print("Open Images CSV IDs:", len(open_images_ids))
print("Matching IDs:", len(matching_ids))
print("Local images without matching IDs:", len(local_ids - open_images_ids))

print("\nSample local filenames:")
for image_id in sorted(local_ids)[:10]:
    print(image_id)

print("\nSample matching IDs:")
for image_id in sorted(matching_ids)[:10]:
    print(image_id)

