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
MODEL_NAME = 'asl_cnn_classifier.h5'
DATA_DIR = r"C:\Users\aniru\OneDrive\Documents\ASL-ENGLISH-Translator\TEST\images"

# ========================
# LOAD MODEL
# ========================
model = tf.keras.models.load_model(MODEL_NAME)

# ========================
# DATA LOADING FUNCTION
# ========================
def create_data(data_dir, start=0, end=200):
    """Loads images and assigns labels based on the first letter of the filename."""
    data = []
    label_set = set()
    
    images = sorted(os.listdir(data_dir))[start:end]
    for img_name in tqdm(images, desc="Loading Data"):
        img_path = os.path.join(data_dir, img_name)
        
        if not os.path.isfile(img_path):
            continue  # Skip non-file items
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue  # Skip if image can't be read
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0  # Normalize
        
        label = img_name[0].upper()  # Extract first letter as label
        label_set.add(label)
        data.append([img, label])  # Append (image, label)
    
    return data, sorted(label_set)

# Load Data
test_data, label_names = create_data(DATA_DIR)
label_to_index = {label: idx for idx, label in enumerate(label_names)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Convert labels to indices
X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_test = np.array([label_to_index[i[1]] for i in test_data])

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

# Classification Report (Fixing the missing label issue)
all_labels = list(range(len(label_names)))  # Ensures all class indices are included

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, labels=all_labels, target_names=label_names, zero_division=0))

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
