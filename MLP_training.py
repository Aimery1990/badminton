import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns  # For visualization of confusion matrix

from Assignment4_DataProcess import run_data_processing_mlp

run_data_processing_mlp()

# Define the classification column name (Assuming the Keywords class is available)
CLASSIFICATION_COLUMN = 'shot_type'

# --- 1. Load Data ---
df_train = pd.read_csv("./combined_final_features/final_combined_training.csv")
df_test = pd.read_csv("./combined_final_features/final_combined_validation.csv")

# Separate features (X) and labels (y)
X_train = df_train.drop(columns=[CLASSIFICATION_COLUMN])
y_train_raw = df_train[CLASSIFICATION_COLUMN]
X_test = df_test.drop(columns=[CLASSIFICATION_COLUMN])
y_test_raw = df_test[CLASSIFICATION_COLUMN]

# --- 2. Label Encoding (for Keras) ---
# Convert string labels (e.g., 'clear', 'drive') to numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_raw)
y_test_encoded = label_encoder.transform(y_test_raw)

# One-hot encode the labels for categorical cross-entropy loss
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

# Get the list of classes for decoding/reporting
class_names = label_encoder.classes_
print(f"Detected Classes: {class_names}")

# --- 3. Build Keras MLP Model ---
input_dim = X_train.shape[1]
output_dim = y_train_one_hot.shape[1]

print(f"Input Features: {input_dim}, Output Classes: {output_dim}")

model = Sequential([
    # Layer 1: 128 neurons, ReLU activation, Dropout
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),  # Implements the requested dropout

    # Layer 2: 64 neurons, ReLU activation, Dropout
    Dense(64, activation='relu'),
    Dropout(0.3),

    # Output Layer: Softmax for multi-class probability output
    Dense(output_dim, activation='softmax')
])

# --- 4. Compile and Train ---
# Loss = CrossEntropyLoss (categorical_crossentropy in Keras), Optimizer = Adam
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model Training Starting ---")
history = model.fit(
    X_train, y_train_one_hot,
    epochs=50,  # Typically requires more epochs than scikit-learn's default
    batch_size=32,
    validation_data=(X_test, y_test_one_hot),
    verbose=2  # Show progress every epoch
)
print("--- Model Training Finished ---")

# --- 5. Evaluate Accuracy, Macro F1, and Confusion Matrix ---

# Predict probabilities
y_pred_probs = model.predict(X_test, verbose=0)
# Convert probabilities to class labels (takes the index with max probability)
y_pred_encoded = np.argmax(y_pred_probs, axis=1)

# Convert predicted integer labels back to original string labels for the report
y_pred_class = label_encoder.inverse_transform(y_pred_encoded)

print("\n--- Classification Report (Accuracy & Macro F1) ---")
print(classification_report(y_test_raw, y_pred_class, target_names=class_names))

print(f"Overall Test Accuracy: {accuracy_score(y_test_raw, y_pred_class):.4f}")

# Calculate and display Confusion Matrix
cm = confusion_matrix(y_test_raw, y_pred_class, labels=class_names)
print("\n--- Confusion Matrix ---")
print(cm)

# Visualize Confusion Matrix for better understanding
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix for MLP ({len(class_names)} Classes)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

def plot_training_curves(history):
    """Plot loss and accuracy curves for Keras model training."""

    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    # Different Keras configs use different accuracy keys
    if "accuracy" in history.history:
        train_acc = history.history["accuracy"]
        val_acc = history.history.get("val_accuracy", [])
    elif "categorical_accuracy" in history.history:
        train_acc = history.history["categorical_accuracy"]
        val_acc = history.history.get("val_categorical_accuracy", [])
    elif "sparse_categorical_accuracy" in history.history:
        train_acc = history.history["sparse_categorical_accuracy"]
        val_acc = history.history.get("val_sparse_categorical_accuracy", [])
    else:
        raise KeyError(
            f"No accuracy key found in history.history: {list(history.history.keys())}"
        )

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    if val_acc:
        plt.plot(epochs, val_acc, label="Validation Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history)