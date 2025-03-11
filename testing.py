import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

# ========================
# CONFIGURATION & SETUP
# ========================
IMG_SIZE = 64
MODEL_NAME = 'asl_cnn_classifier_no_aug.h5'
DATA_DIR = r"C:\Users\lenovo\Documents\GitHub\ASL-Translator\ASLYset\ASLYset\images\User3"
MAX_IMAGES_PER_CLASS = 10

# ========================
# LOAD MODEL
# ========================
model = tf.keras.models.load_model(MODEL_NAME)

def load_labels():
    """Loads label names from the folder structure."""
    return sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])

# Initialize label mapping
label_names = load_labels()
label_to_index = {label: idx for idx, label in enumerate(label_names)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

num_classes = len(label_to_index)

# ========================
# IMAGE PROCESSING
# ========================
def process_image(img_path):
    """Reads, resizes, and normalizes an image."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return img

def create_data(data_dir):
    """Loads images and labels from a directory structure, excluding 'sp' and 'fn' folders."""
    data = []
    labels = []
    all_labels = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    # Exclude 'sp' and 'fn' folders
    all_labels = [label for label in all_labels if label not in ["sp", "fn"]]

    for label in tqdm(all_labels, desc="Loading Data (Folders)"):
        label_path = os.path.join(data_dir, label)
        
        if label not in label_to_index:
            continue  # Skip unknown labels

        images = sorted(os.listdir(label_path))[len(os.listdir(label_path)) - 1:len(os.listdir(label_path)) - MAX_IMAGES_PER_CLASS - 1:-1]
        img_paths = [os.path.join(label_path, img_name) for img_name in images if os.path.isfile(os.path.join(label_path, img_name))]

        for img_path in img_paths:
            processed = process_image(img_path)
            if processed is not None:
                data.append(processed)
                labels.append(label_to_index[label])

    return np.array(data), np.array(labels)

# Load Data
X_test, Y_test = create_data(DATA_DIR)

# ========================
# MODEL EVALUATION
# ========================
Y_pred_probs = model.predict(X_test)
Y_pred = np.argmax(Y_pred_probs, axis=1)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='macro', zero_division=0)
recall = recall_score(Y_test, Y_pred, average='macro', zero_division=0)
f1 = f1_score(Y_test, Y_pred, average='macro', zero_division=0)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Debugging: Check missing classes
print(f"Labels in test set: {sorted(set(Y_test))}")
print(f"Expected labels: {list(range(len(label_names)))}")

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, labels=list(label_to_index.values()), target_names=label_names, zero_division=0))

# ========================
# VISUALIZATION
# ========================
def plot_predictions(images, true_labels, pred_labels):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(min(9, len(images))):
        img = images[i]
        true_label = index_to_label[true_labels[i]]
        pred_label = index_to_label[pred_labels[i]]
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_predictions(X_test[:9], Y_test[:9], Y_pred[:9])
