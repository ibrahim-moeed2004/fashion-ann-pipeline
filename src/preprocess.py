from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PARAMS_PATH = Path("params.yaml")


def main():
    # Load parameters
    with open(PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    val_size = params["preprocess"]["val_size"]
    seed = params["preprocess"]["seed"]

    # Load raw data
    x_train = np.load(RAW_DIR / "x_train.npy")
    y_train = np.load(RAW_DIR / "y_train.npy")
    x_test = np.load(RAW_DIR / "x_test.npy")
    y_test = np.load(RAW_DIR / "y_test.npy")

    # Normalize images
    x_train = (x_train.astype("float32") / 255.0) - 0.5
    x_test = (x_test.astype("float32") / 255.0) - 0.5


    # Create training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train
    )

    # Create processed data directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed arrays
    np.save(PROCESSED_DIR / "x_train.npy", x_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "x_val.npy", x_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "x_test.npy", x_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print("Fashion-MNIST preprocessing completed successfully.")
    print(f"Training images: {x_train.shape}")
    print(f"Validation images: {x_val.shape}")
    print(f"Test images: {x_test.shape}")
    print(f"Pixel range: {x_train.min():.1f} to {x_train.max():.1f}")


main()