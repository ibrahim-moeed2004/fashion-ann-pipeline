from pathlib import Path
import csv

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
PARAMS_PATH = Path("params.yaml")


def main():
    # Load parameters
    with open(PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    train_params = params["train"]

    seed = train_params["seed"]
    dense_units = train_params["dense_units"]
    dropout_rate = train_params["dropout_rate"]
    learning_rate = train_params["learning_rate"]
    epochs = train_params["epochs"]
    batch_size = train_params["batch_size"]

    # Set reproducibility
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    # Load processed data
    x_train = np.load(PROCESSED_DIR / "x_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    x_val = np.load(PROCESSED_DIR / "x_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    # Build ANN model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(10, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    model.save(MODEL_DIR / "model.h5")

    # Save training history
    history_path = MODEL_DIR / "history.csv"

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(history.history.keys())
        writer.writerows(zip(*history.history.values()))

    print("\nTraining completed successfully.")
    print("Model saved to: models/model.h5")
    print("Training history saved to: models/history.csv")


main()