from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist


# Output Directory
RAW_DIR = Path("data/raw")


def main():
    # Create raw data directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load Fashion-MNIST
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Save raw arrays
    np.save(RAW_DIR / "x_train.npy", x_train)
    np.save(RAW_DIR / "y_train.npy", y_train)
    np.save(RAW_DIR / "x_test.npy", x_test)
    np.save(RAW_DIR / "y_test.npy", y_test)

    print("Fashion-MNIST downloaded and saved successfully.")
    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")


main()