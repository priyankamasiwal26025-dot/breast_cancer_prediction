import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = "data"

def evaluate_model():
    model = tf.keras.models.load_model("model.h5")

    test_datagen = ImageDataGenerator(rescale=1/255.)

    test = test_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    loss, acc = model.evaluate(test)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")

if __name__ == "__main__":
    evaluate_model()
