from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.models import load_model


# Paths
PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/model.h5")
CONFUSION_MATRIX_PATH = Path("models/confusion_matrix.png")
METRICS_PATH = Path("metrics.json")


def main():
    # Load test data
    x_test = np.load(PROCESSED_DIR / "x_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    # Load trained model
    model = load_model(MODEL_PATH)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    # Generate predictions
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model evaluation completed successfully.")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Confusion matrix saved to: models/confusion_matrix.png")
    print("Metrics saved to: metrics.json")


main()