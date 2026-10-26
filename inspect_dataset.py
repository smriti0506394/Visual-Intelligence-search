import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "data" / "annotations"

files = [
    "validation-images-with-rotation.csv",
    "validation-annotations-bbox.csv",
    "oidv7-class-descriptions-boxable.csv",
]

headers = {}

for filename in files:
    path = ANNOTATIONS / filename

    print(f"\n{'=' * 60}")
    print(f"File: {filename}")

    if not path.exists():
        print("ERROR: File not found!")
        continue

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers[filename] = reader.fieldnames or []
        print("Headers:", headers[filename])

        for row_number, row in enumerate(reader, start=1):
            if row_number <= 2:
                print(f"Sample row {row_number}:", row)

        print("Total data rows:", row_number if 'row_number' in locals() else 0)
        # The row count above is refined in the next step if needed.

# Compare image IDs only when both files contain ImageID.
image_file = "validation-images-with-rotation.csv"
bbox_file = "validation-annotations-bbox.csv"

if (
    image_file in headers
    and bbox_file in headers
    and "ImageID" in headers[image_file]
    and "ImageID" in headers[bbox_file]
):
    def read_ids(filename):
        path = ANNOTATIONS / filename
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return {
                row["ImageID"]
                for row in csv.DictReader(f)
                if row.get("ImageID")
            }

    image_ids = read_ids(image_file)
    bbox_ids = read_ids(bbox_file)

    overlap = image_ids & bbox_ids

    print(f"\n{'=' * 60}")
    print("IMAGE ID COMPATIBILITY CHECK")
    print("Unique image IDs in image CSV:", len(image_ids))
    print("Unique image IDs in bounding-box CSV:", len(bbox_ids))
    print("Matching image IDs:", len(overlap))
    print("Image IDs without bounding-box annotations:", len(image_ids - bbox_ids))
    print("Annotation IDs without image-list entries:", len(bbox_ids - image_ids))
else:
    print("\nCould not compare IDs: check the printed headers.")