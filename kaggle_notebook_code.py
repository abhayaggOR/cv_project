import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

IMAGE_PATHS = [
    "/kaggle/input/datasets/abhay1470/testim/new10.jpg",
    "/kaggle/input/datasets/abhay1470/testim/new100.jpg",
    "/kaggle/input/datasets/abhay1470/testim/new110.jpg",
    "/kaggle/input/datasets/abhay1470/testim/new128.jpg",
]

all_results = []

for i, image_path in enumerate(IMAGE_PATHS, 1):
    print(f"\n===== Image {i}/{len(IMAGE_PATHS)} =====")
    print("Path:", image_path)

    if not Path(image_path).exists():
        print("File not found.")
        all_results.append({
            "image_path": image_path,
            "error": "File not found"
        })
        continue

    try:
        result = predict_assignment_json(image_path, vehicle_conf=0.15, hp_conf=0.10)
        all_results.append({
            "image_path": image_path,
            "result": result
        })

        print("Assignment JSON output:")
        print(json.dumps(result, indent=2))

        img = cv2.imread(image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 7))
            plt.imshow(img_rgb)
            plt.title(f"{Path(image_path).name} | violations: {len(result['violations'])}")
            plt.axis("off")
            plt.show()

    except Exception as e:
        print("Error:", str(e))
        all_results.append({
            "image_path": image_path,
            "error": str(e)
        })

out_path = "/kaggle/working/test_results_4images.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved results to:", out_path)
import json
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import easyocr
from ultralytics import YOLO

# ---------- model paths ----------
custom_model_path_candidates = [
    "/kaggle/working/exported_models/helmet_plate_best.pt",
    "/kaggle/working/helmet_plate_yolo11n/weights/best.pt",
]

custom_model_path = None
for p in custom_model_path_candidates:
    if Path(p).exists():
        custom_model_path = p
        break

if custom_model_path is None:
    raise FileNotFoundError(
        "Could not find your trained custom model. Expected one of:\n" +
        "\n".join(custom_model_path_candidates)
    )

print("Using custom model:", custom_model_path)

# ---------- load models ----------
vehicle_model = YOLO("yolo11n.pt")                # COCO pretrained
helmet_plate_model = YOLO(custom_model_path)     # your trained model
ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# ---------- helpers ----------
def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")

def clip_int(v, lo, hi):
    return int(max(lo, min(hi, v)))

def expand_box(box, img_w, img_h, top=0.9, sides=0.2, bottom=0.15):
    x1, y1, x2, y2 = map(float, box)
    w = x2 - x1
    h = y2 - y1
    return np.array([
        clip_int(x1 - sides * w, 0, img_w - 1),
        clip_int(y1 - top * h, 0, img_h - 1),
        clip_int(x2 + sides * w, 0, img_w - 1),
        clip_int(y2 + bottom * h, 0, img_h - 1),
    ], dtype=int)

def center_in_region(box, region):
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = region
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

def sanitize_plate(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

def ocr_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    texts = ocr_reader.readtext(
        thr,
        detail=0,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    texts = [sanitize_plate(t) for t in texts]
    texts = [t for t in texts if len(t) >= 4]
    return max(texts, key=len, default="")

# ---------- main prediction ----------
def predict_assignment_json(image_path, vehicle_conf=0.15, hp_conf=0.10):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    H, W = img.shape[:2]

    det = vehicle_model.predict(img, conf=vehicle_conf, verbose=False)[0]
    coco_names = det.names

    people = []
    bikes = []

    if det.boxes is not None and len(det.boxes) > 0:
        xyxy = det.boxes.xyxy.cpu().numpy()
        cls = det.boxes.cls.cpu().numpy().astype(int)

        for box, cls_id in zip(xyxy, cls):
            cls_name = coco_names[int(cls_id)]
            if cls_name == "person":
                people.append(box)
            elif cls_name == "motorcycle":
                bikes.append(box)

    output = {"violations": []}

    for bike_box in bikes:
        region = expand_box(bike_box, W, H)
        rx1, ry1, rx2, ry2 = region
        bike_crop = img[ry1:ry2, rx1:rx2].copy()

        num_riders = sum(center_in_region(p, region) for p in people)

        crop_det = helmet_plate_model.predict(bike_crop, conf=hp_conf, verbose=False)[0]
        hp_names = crop_det.names

        nohelmet_boxes = []
        plate_boxes = []

        if crop_det.boxes is not None and len(crop_det.boxes) > 0:
            xyxy = crop_det.boxes.xyxy.cpu().numpy()
            cls = crop_det.boxes.cls.cpu().numpy().astype(int)

            for box, cls_id in zip(xyxy, cls):
                name = normalize_name(hp_names[int(cls_id)])
                if "withouthelmet" in name or "nohelmet" in name:
                    nohelmet_boxes.append(box)
                elif "plate" in name or "licence" in name or "license" in name:
                    plate_boxes.append(box)

        helmet_violations = len(nohelmet_boxes)

        if num_riders == 0 and helmet_violations > 0:
            num_riders = helmet_violations

        num_riders = max(num_riders, 1)
        helmet_violations = min(helmet_violations, num_riders)

        if num_riders > 2 or helmet_violations > 0:
            plate_text = ""
            if plate_boxes:
                best_plate = max(
                    plate_boxes,
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
                )
                px1, py1, px2, py2 = map(int, best_plate)
                plate_crop = bike_crop[py1:py2, px1:px2]
                plate_text = ocr_plate(plate_crop)

            output["violations"].append({
                "num_riders": int(num_riders),
                "helmet_violations": int(helmet_violations),
                "license_plate": plate_text
            })

    return output

# ---------- test images ----------
IMAGE_PATHS = [
    "/kaggle/input/datasets/abhay1470/testim/new10.jpg",
    "/kaggle/input/datasets/abhay1470/testim/new100.jpg",
    "/kaggle/input/datasets/abhay1470/testim/new110.jpg",
    "/kaggle/input/datasets/abhay1470/testim/new128.jpg",
]

all_results = []

for i, image_path in enumerate(IMAGE_PATHS, 1):
    print(f"\n===== Image {i}/{len(IMAGE_PATHS)} =====")
    print("Path:", image_path)

    if not Path(image_path).exists():
        print("File not found.")
        all_results.append({"image_path": image_path, "error": "File not found"})
        continue

    try:
        result = predict_assignment_json(image_path, vehicle_conf=0.15, hp_conf=0.10)
        all_results.append({
            "image_path": image_path,
            "result": result
        })

        print(json.dumps(result, indent=2))

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 7))
        plt.imshow(img_rgb)
        plt.title(f"{Path(image_path).name} | violations: {len(result['violations'])}")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print("Error:", str(e))
        all_results.append({
            "image_path": image_path,
            "error": str(e)
        })

out_path = "/kaggle/working/test_results_4images.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved results to:", out_path)
!pip -q install ultralytics easyocr kagglehub
import os
import re
import json
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import kagglehub

DATASET_SLUG = "pkdarabi/helmet"  # HelmetViolations: Plate / WithHelmet / WithoutHelmet

dataset_root = Path(kagglehub.dataset_download(DATASET_SLUG))
print("Downloaded to:", dataset_root)

def find_first_yolo_yaml(root: Path):
    yamls = list(root.rglob("*.yaml"))
    if not yamls:
        raise FileNotFoundError("No YAML file found inside dataset")

    print("\nYAML files found:")
    for y in yamls[:20]:
        print(" -", y)

    for y in yamls:
        txt = y.read_text(errors="ignore").lower()
        if "train:" in txt and "val:" in txt and "names:" in txt:
            return y
    return yamls[0]

data_yaml = find_first_yolo_yaml(dataset_root)
print("\nUsing data yaml:", data_yaml)
print("\n--- YAML preview ---")
print(data_yaml.read_text()[:2000])

def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

all_images = list_images(dataset_root)
print(f"\nFound {len(all_images)} images")
print(all_images[:10])
sample_imgs = all_images[:9]

plt.figure(figsize=(14, 14))
for i, img_path in enumerate(sample_imgs, 1):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, i)
    plt.imshow(img)
    plt.title(img_path.name)
    plt.axis("off")
plt.tight_layout()
plt.show()
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data=str(data_yaml),
    epochs=40,
    imgsz=640,
    batch=16,
    device=0,
    optimizer="AdamW",
    lr0=1e-3,
    patience=10,
    cache=True,
    pretrained=True,
    project="/kaggle/working",
    name="helmet_plate_yolo11n",
    mosaic=1.0,
    mixup=0.1,
    degrees=5.0,
    scale=0.4,
    fliplr=0.5,
    workers=4,
)

best_path = Path("/kaggle/working/helmet_plate_yolo11n/weights/best.pt")
print("Best weights:", best_path)
print("Exists:", best_path.exists())
best_model = YOLO(str(best_path))
metrics = best_model.val(data=str(data_yaml), split="val")
print(metrics.results_dict)
export_dir = Path("/kaggle/working/exported_models")
export_dir.mkdir(parents=True, exist_ok=True)

final_weight = export_dir / "helmet_plate_best.pt"
shutil.copy(best_path, final_weight)

print("Saved to:", final_weight)
import torch
import easyocr
from ultralytics import YOLO

vehicle_model = YOLO("yolo11n.pt")           # COCO pretrained: person + motorcycle
helmet_plate_model = YOLO(str(best_path))    # your trained model
ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")

def clip_int(v, lo, hi):
    return int(max(lo, min(hi, v)))

def expand_box(box, img_w, img_h, top=0.9, sides=0.2, bottom=0.15):
    x1, y1, x2, y2 = map(float, box)
    w = x2 - x1
    h = y2 - y1
    return np.array([
        clip_int(x1 - sides * w, 0, img_w - 1),
        clip_int(y1 - top * h, 0, img_h - 1),
        clip_int(x2 + sides * w, 0, img_w - 1),
        clip_int(y2 + bottom * h, 0, img_h - 1),
    ], dtype=int)

def center_in_region(box, region):
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = region
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

def sanitize_plate(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

def ocr_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    texts = ocr_reader.readtext(
        thr,
        detail=0,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    texts = [sanitize_plate(t) for t in texts]
    texts = [t for t in texts if len(t) >= 5]
    return max(texts, key=len, default="")

def predict_assignment_json(image_path, vehicle_conf=0.25, hp_conf=0.20):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    H, W = img.shape[:2]

    det = vehicle_model.predict(img, conf=vehicle_conf, verbose=False)[0]
    coco_names = det.names

    people = []
    bikes = []

    if det.boxes is not None and len(det.boxes) > 0:
        xyxy = det.boxes.xyxy.cpu().numpy()
        cls = det.boxes.cls.cpu().numpy().astype(int)

        for box, cls_id in zip(xyxy, cls):
            cls_name = coco_names[int(cls_id)]
            if cls_name == "person":
                people.append(box)
            elif cls_name == "motorcycle":
                bikes.append(box)

    output = {"violations": []}

    for bike_box in bikes:
        region = expand_box(bike_box, W, H)
        rx1, ry1, rx2, ry2 = region
        bike_crop = img[ry1:ry2, rx1:rx2].copy()

        num_riders = sum(center_in_region(p, region) for p in people)

        crop_det = helmet_plate_model.predict(bike_crop, conf=hp_conf, verbose=False)[0]
        hp_names = crop_det.names

        nohelmet_boxes = []
        plate_boxes = []

        if crop_det.boxes is not None and len(crop_det.boxes) > 0:
            xyxy = crop_det.boxes.xyxy.cpu().numpy()
            cls = crop_det.boxes.cls.cpu().numpy().astype(int)

            for box, cls_id in zip(xyxy, cls):
                name = normalize_name(hp_names[int(cls_id)])
                if "withouthelmet" in name or "nohelmet" in name:
                    nohelmet_boxes.append(box)
                elif "plate" in name or "licence" in name or "license" in name:
                    plate_boxes.append(box)

        helmet_violations = len(nohelmet_boxes)

        if num_riders == 0 and helmet_violations > 0:
            num_riders = helmet_violations
        num_riders = max(num_riders, 1)
        helmet_violations = min(helmet_violations, num_riders)

        if num_riders > 2 or helmet_violations > 0:
            plate_text = ""
            if plate_boxes:
                best_plate = max(
                    plate_boxes,
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
                )
                px1, py1, px2, py2 = map(int, best_plate)
                plate_crop = bike_crop[py1:py2, px1:px2]
                plate_text = ocr_plate(plate_crop)

            output["violations"].append({
                "num_riders": int(num_riders),
                "helmet_violations": int(helmet_violations),
                "license_plate": plate_text
            })

    return output
# Auto-pick one sample image from the downloaded dataset
valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
candidate_images = [p for p in dataset_root.rglob("*") if p.suffix.lower() in valid_exts]

if not candidate_images:
    raise FileNotFoundError("No images found in dataset_root")

TEST_IMAGE = str(candidate_images[0])
print("Using test image:", TEST_IMAGE)

result = predict_assignment_json(TEST_IMAGE)
print(json.dumps(result, indent=2))
def show_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

show_image(TEST_IMAGE)
from kaggle_secrets import UserSecretsClient
from urllib.parse import quote
import subprocess

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
auth_url = f"https://abhayaggOR:{quote(token)}@github.com/abhayaggOR/cv_project.git"

print("Token loaded:", token[:12], "...")

subprocess.run(["git", "ls-remote", auth_url], check=True)
print("GitHub auth looks okay.")
from kaggle_secrets import UserSecretsClient
from pathlib import Path
from urllib.parse import quote
import subprocess
import shutil
import json
import os

# -------- settings --------
GITHUB_USER = "abhayaggOR"
REPO_NAME = "cv_project"
BRANCH = "main"
RUN_NAME = "kaggle_run_01"   # change per run if you want
REPO_DIR = Path("/kaggle/working/repo")
ARTIFACT_DIR = REPO_DIR / "kaggle_artifacts" / RUN_NAME

# files you want to push
BEST_MODEL = Path("/kaggle/working/exported_models/helmet_plate_best.pt")
TRAIN_DIR = Path("/kaggle/working/helmet_plate_yolo11n")
RESULTS_CSV = TRAIN_DIR / "results.csv"
CONFUSION = TRAIN_DIR / "confusion_matrix.png"
PR_CURVE = TRAIN_DIR / "PR_curve.png"

# -------- auth --------
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
auth_url = f"https://{GITHUB_USER}:{quote(token)}@github.com/{GITHUB_USER}/{REPO_NAME}.git"

# -------- clone repo --------
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

subprocess.run(["git", "clone", auth_url, str(REPO_DIR)], check=True)

# -------- git identity --------
subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.name", "Kaggle Bot"], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.email", "kaggle-bot@example.com"], check=True)

# -------- prepare artifact folder --------
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

copied = []

def safe_copy(src, dst_dir):
    src = Path(src)
    if src.exists():
        dst = Path(dst_dir) / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))

safe_copy(BEST_MODEL, ARTIFACT_DIR)
safe_copy(RESULTS_CSV, ARTIFACT_DIR)
safe_copy(CONFUSION, ARTIFACT_DIR)
safe_copy(PR_CURVE, ARTIFACT_DIR)

# optional: save a small summary json
summary = {
    "run_name": RUN_NAME,
    "copied_files": copied,
}
with open(ARTIFACT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# -------- commit & push --------
subprocess.run(["git", "-C", str(REPO_DIR), "add", "."], check=True)

commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "commit", "-m", f"Add Kaggle artifacts: {RUN_NAME}"],
    capture_output=True,
    text=True
)

print(commit.stdout)
print(commit.stderr)

# commit may fail if nothing changed; push only if commit succeeded
if commit.returncode == 0:
    subprocess.run(["git", "-C", str(REPO_DIR), "push", "origin", BRANCH], check=True)
    print("Pushed to GitHub successfully.")
else:
    print("No new changes to commit.")
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

img = cv2.imread(TEST_IMAGE)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1) COCO motorcycle/person detections
det1 = vehicle_model.predict(img, conf=0.15, verbose=False)[0]
vis1 = det1.plot()

# 2) Helmet/plate detections
det2 = helmet_plate_model.predict(img, conf=0.10, verbose=False)[0]
vis2 = det2.plot()

plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
plt.title("COCO: person + motorcycle")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
plt.title("Custom: helmet / no helmet / plate")
plt.axis("off")

plt.tight_layout()
plt.show()
valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
candidate_images = [p for p in dataset_root.rglob("*") if p.suffix.lower() in valid_exts]

found = None

for p in candidate_images[:100]:
    result = predict_assignment_json(str(p), vehicle_conf=0.15, hp_conf=0.10)
    if result["violations"]:
        found = (str(p), result)
        break

if found is None:
    print("No violating image found in first 100 images.")
else:
    TEST_IMAGE = found[0]
    print("Found candidate:", TEST_IMAGE)
    print(json.dumps(found[1], indent=2))
metrics = best_model.val(data=str(data_yaml), split="val")
print(metrics.results_dict)
import json
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import easyocr

# ---------- path ----------
CUSTOM_IMAGE = "/kaggle/input/datasets/abhay1470/image02"

p = Path(CUSTOM_IMAGE)
if not p.exists():
    raise FileNotFoundError(f"Path not found: {CUSTOM_IMAGE}")

if p.is_dir():
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted([x for x in p.rglob("*") if x.suffix.lower() in valid_exts])
    if not imgs:
        raise FileNotFoundError(f"No image files found inside directory: {CUSTOM_IMAGE}")
    CUSTOM_IMAGE = str(imgs[0])
    print("Given path is a folder. Using first image found:", CUSTOM_IMAGE)
else:
    print("Using image:", CUSTOM_IMAGE)

# ---------- OCR setup ----------
if "ocr_reader" not in globals():
    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")

def sanitize_plate(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

def ocr_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    texts = ocr_reader.readtext(
        thr,
        detail=0,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    texts = [sanitize_plate(t) for t in texts]
    texts = [t for t in texts if len(t) >= 4]
    return max(texts, key=len, default="")

# ---------- load image ----------
img = cv2.imread(CUSTOM_IMAGE)
if img is None:
    raise FileNotFoundError(f"OpenCV could not read image: {CUSTOM_IMAGE}")

# ---------- assignment pipeline output ----------
result = predict_assignment_json(CUSTOM_IMAGE, vehicle_conf=0.15, hp_conf=0.10)
print("\nAssignment JSON output:")
print(json.dumps(result, indent=2))

# ---------- full-image detections ----------
det1 = vehicle_model.predict(img, conf=0.15, verbose=False)[0]
det2 = helmet_plate_model.predict(img, conf=0.10, verbose=False)[0]

vis1 = det1.plot()
vis2 = det2.plot()

# ---------- direct number plate read ----------
plate_results = []
plate_vis = img.copy()

if det2.boxes is not None and len(det2.boxes) > 0:
    xyxy = det2.boxes.xyxy.cpu().numpy()
    cls = det2.boxes.cls.cpu().numpy().astype(int)
    confs = det2.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(xyxy, cls, confs):
        cls_name = normalize_name(det2.names[int(cls_id)])

        if "plate" in cls_name or "licence" in cls_name or "license" in cls_name:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            plate_crop = img[y1:y2, x1:x2].copy()
            plate_text = ocr_plate(plate_crop)

            plate_results.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(conf),
                "text": plate_text
            })

            label = plate_text if plate_text else "plate"
            cv2.rectangle(plate_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                plate_vis,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

print("\nDirect number plate OCR results:")
if plate_results:
    print(json.dumps(plate_results, indent=2))
else:
    print("No plate detected by the custom model on this image.")

# ---------- show model outputs ----------
plt.figure(figsize=(20, 12))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
plt.title("COCO model: person + motorcycle")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
plt.title("Custom model: helmet / no helmet / plate")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(plate_vis, cv2.COLOR_BGR2RGB))
plt.title("Detected number plate(s) with OCR text")
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------- show cropped plates ----------
if plate_results:
    plt.figure(figsize=(4 * len(plate_results), 4))
    for i, pr in enumerate(plate_results, 1):
        x1, y1, x2, y2 = pr["box"]
        crop = img[y1:y2, x1:x2]
        plt.subplot(1, len(plate_results), i)
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title(pr["text"] if pr["text"] else "plate")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
from pathlib import Path

dataset_root = Path("/kaggle/input/datasets/pkdarabi/helmet")
root = dataset_root / "HelmetViolationsV2"

splits = ["train", "valid", "test"]
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

grand_total_images = 0
grand_total_labels = 0
grand_total_rows = 0

print("Dataset root:", root)
print()

for split in splits:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"

    images = [p for p in img_dir.rglob("*") if p.suffix.lower() in valid_exts] if img_dir.exists() else []
    label_files = list(lbl_dir.rglob("*.txt")) if lbl_dir.exists() else []

    row_count = 0
    for lf in label_files:
        text = lf.read_text().strip()
        if text:
            row_count += len(text.splitlines())

    grand_total_images += len(images)
    grand_total_labels += len(label_files)
    grand_total_rows += row_count

    print(f"{split.upper()}:")
    print("  images:", len(images))
    print("  label files:", len(label_files))
    print("  annotation rows:", row_count)
    print()

print("TOTAL:")
print("  images:", grand_total_images)
print("  label files:", grand_total_labels)
print("  annotation rows:", grand_total_rows)
from kaggle_secrets import UserSecretsClient
from urllib.parse import quote
from pathlib import Path
import subprocess
import shutil
import os

# ---------- settings ----------
GITHUB_USER = "abhayaggOR"
REPO_NAME = "cv_project"
BRANCH = "main"
COMMIT_MSG = "Push Kaggle notebook code and outputs"

REPO_DIR = Path("/kaggle/working/repo_clone")

# Save current notebook input history as a Python file
NOTEBOOK_CODE_FILE = Path("/kaggle/working/kaggle_notebook_code.py")
get_ipython().run_line_magic("history", f"-f {NOTEBOOK_CODE_FILE}")

# Add here whatever you want to push
FILES_TO_PUSH = [
    "/kaggle/working/kaggle_notebook_code.py",
    "/kaggle/working/exported_models/helmet_plate_best.pt",
]

DIRS_TO_PUSH = [
    "/kaggle/working/helmet_plate_yolo11n",   # training outputs
    "/kaggle/working/exported_models",        # exported model folder
]

# ---------- auth ----------
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
auth_url = f"https://{GITHUB_USER}:{quote(token)}@github.com/{GITHUB_USER}/{REPO_NAME}.git"

# ---------- fresh clone ----------
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

subprocess.run(["git", "clone", auth_url, str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.name", "Kaggle Bot"], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.email", "kaggle-bot@example.com"], check=True)

# ---------- copy files ----------
def copy_file(src_path, repo_dir):
    src = Path(src_path)
    if src.exists() and src.is_file():
        dst = repo_dir / src.name
        shutil.copy2(src, dst)
        print("Copied file:", src, "->", dst)
    else:
        print("Skipped missing file:", src)

def copy_dir(src_path, repo_dir):
    src = Path(src_path)
    if src.exists() and src.is_dir():
        dst = repo_dir / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("Copied dir :", src, "->", dst)
    else:
        print("Skipped missing dir :", src)

for f in FILES_TO_PUSH:
    copy_file(f, REPO_DIR)

for d in DIRS_TO_PUSH:
    copy_dir(d, REPO_DIR)

# ---------- git add / commit / push ----------
subprocess.run(["git", "-C", str(REPO_DIR), "add", "."], check=True)

commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "commit", "-m", COMMIT_MSG],
    capture_output=True,
    text=True
)

print(commit.stdout)
print(commit.stderr)

if commit.returncode == 0:
    subprocess.run(["git", "-C", str(REPO_DIR), "push", "origin", BRANCH], check=True)
    print("Push successful.")
else:
    print("No new changes to commit.")
