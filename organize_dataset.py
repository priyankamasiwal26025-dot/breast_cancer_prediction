import os
import shutil
import random

# Your actual source folders
SOURCE_DIR = "data"   # your data folder
SOURCE_BENIGN = os.path.join(SOURCE_DIR, "Benign")
SOURCE_MALIGNANT = os.path.join(SOURCE_DIR, "Malignant")

# Target folders
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Create required folders
for path in [
    TRAIN_DIR, TEST_DIR,
    os.path.join(TRAIN_DIR, "benign"),
    os.path.join(TRAIN_DIR, "malignant"),
    os.path.join(TEST_DIR, "benign"),
    os.path.join(TEST_DIR, "malignant"),
]:
    os.makedirs(path, exist_ok=True)

def split_and_move(source_folder, class_name):
    images = [f for f in os.listdir(source_folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    # 80% train, 20% test
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Move train images
    for img in train_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(TRAIN_DIR, class_name, img)
        )

    # Move test images
    for img in test_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(TEST_DIR, class_name, img)
        )

    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

# Run splitting
split_and_move(SOURCE_BENIGN, "benign")
split_and_move(SOURCE_MALIGNANT, "malignant")

print("Dataset organized successfully!")
