import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ===========================
#  CONFIG
# ===========================
DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_SAVE_PATH = "model.h5"   # final saved model

# ===========================
#  LOAD DATASET
# ===========================
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

print("Loading dataset...")
print("Train directory:", train_dir)
print("Test directory:", test_dir)

train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Better performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ===========================
#  MODEL
# ===========================
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')   # binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===========================
#  TRAIN MODEL
# ===========================
print("\nTraining model...")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# ===========================
#  SAVE MODEL
# ===========================
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved successfully → {MODEL_SAVE_PATH}")
