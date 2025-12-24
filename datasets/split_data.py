import os
import csv
import random

ROOT_DIR = "datasets/archive/256_ObjectCategories"
OUT_DIR = "datasets/splits"

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)

image_exts = (".jpg", ".jpeg", ".png", ".bmp")

data = []
label2id = {}

# =========================
# Duyệt dataset
# =========================
for label_id, class_name in enumerate(sorted(os.listdir(ROOT_DIR))):
    class_path = os.path.join(ROOT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    label2id[class_name] = label_id

    for fname in os.listdir(class_path):
        if fname.lower().endswith(image_exts):
            # 🔥 chỉ lưu path tương đối
            rel_path = os.path.join(class_name, fname)

            data.append({
                "image_path": rel_path,
                "label_name": class_name,
                "label_id": label_id
            })

print(f"Total images: {len(data)}")
print(f"Total classes: {len(label2id)}")

# =========================
# Shuffle + Split
# =========================
random.shuffle(data)

n_total = len(data)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)

train_data = data[:n_train]
val_data   = data[n_train:n_train + n_val]
test_data  = data[n_train + n_val:]

print(f"Train: {len(train_data)}")
print(f"Val  : {len(val_data)}")
print(f"Test : {len(test_data)}")

# =========================
# Save CSV
# =========================
def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label_name", "label_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

save_csv(os.path.join(OUT_DIR, "metadata.csv"), data)
save_csv(os.path.join(OUT_DIR, "train.csv"), train_data)
save_csv(os.path.join(OUT_DIR, "val.csv"), val_data)
save_csv(os.path.join(OUT_DIR, "test.csv"), test_data)

print("\nSaved:")
print(" - splits/metadata.csv")
print(" - splits/train.csv")
print(" - splits/val.csv")
print(" - splits/test.csv")
