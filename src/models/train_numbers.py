from __future__ import annotations

import json
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import BATCH_SIZE, EPOCHS, IMG_SIZE, NUMBER_MODEL, NUMBERS_TRAIN_DIR


def train_number_model(data_dir: Path, model_path: Path) -> None:
    print("Loading number images...")
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
    )

    train_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    num_classes = len(train_gen.class_indices)
    print(f"Classes found: {num_classes}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_index_path = model_path.with_name("number_classes.json")
    with class_index_path.open("w", encoding="utf-8") as file_handle:
        json.dump(train_gen.class_indices, file_handle, indent=2)
    print(f"Saved class indices to: {class_index_path}")

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print("Training number CNN...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(model_path), save_best_only=True, verbose=1),
        ],
    )

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_number_model(Path(NUMBERS_TRAIN_DIR), Path(NUMBER_MODEL))
