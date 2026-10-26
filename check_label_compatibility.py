
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "data" / "annotations"

bbox_file = ANNOTATIONS / "validation-annotations-bbox.csv"
classes_file = ANNOTATIONS / "oidv7-class-descriptions-boxable.csv"

# Read unique label IDs from bounding-box annotations
bbox_labels = set()

with bbox_file.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("LabelName"):
            bbox_labels.add(row["LabelName"])

# Read label IDs and their human-readable names
class_names = {}

with classes_file.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        label = row.get("LabelName")
        name = row.get("DisplayName")

        if label and name:
            class_names[label] = name

matched = bbox_labels & class_names.keys()
unmatched = bbox_labels - class_names.keys()

print("=" * 55)
print("LABEL COMPATIBILITY REPORT")
print("=" * 55)

print("Unique labels in bounding-box CSV:", len(bbox_labels))
print("Unique labels in class-description CSV:", len(class_names))
print("Labels with matching names:", len(matched))
print("Labels without a class-name mapping:", len(unmatched))

print("\nSample matched labels:")
for label in sorted(matched)[:10]:
    print(f"{label} -> {class_names[label]}")

print("\nSample unmatched labels:")
for label in sorted(unmatched)[:20]:
    print(label)

if not unmatched:
    print("\nAll annotation labels have a class-name mapping.")
else:
    print(
        "\nSome annotation labels have no mapping in the V7 "
        "class-description file. Do not assume these labels "
        "are compatible until we investigate them."
    )
