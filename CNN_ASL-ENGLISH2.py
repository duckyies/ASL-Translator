import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle

# ========================
# CONFIGURATION & SETUP
# ========================
IMG_SIZE = 64
MODEL_NAME = 'asl_cnn_classifier_no_aug.h5'
MAX_IMAGES_PER_CLASS = 40  # Limit number of images per class for faster loading

TRAIN_DIR = r"C:\Users\lenovo\Documents\GitHub\ASL-Translator\ASLYset\ASLYset\images\User1"

# ========================
# DATA LOADING & LABEL HANDLING
# ========================
def load_labels():
    """Loads label names from folder-based dataset."""
    return sorted([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])

# Initialize label mapping
label_names = load_labels()
label_to_index = {label: idx for idx, label in enumerate(label_names)}
index_to_label = {idx: label for idx, label in enumerate(label_names)}

def update_labels_from_filenames(data_dir):
    """Dynamically update label_to_index to include missing labels from filename-based dataset."""
    global label_to_index, index_to_label
    for img_name in os.listdir(data_dir):
        label = img_name[0].upper()  # Extract label from filename
        if label not in label_to_index:
            new_index = len(label_to_index)
            label_to_index[label] = new_index
            index_to_label[new_index] = label

# Update label mapping using filename-based dataset
update_labels_from_filenames(TRAIN_DIR)

num_classes = len(label_to_index)

# ========================
# IMAGE PROCESSING
# ========================
def process_image(img_path, label_index):
    """Reads, resizes, and normalizes an image."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (optional but useful)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return [img, label_index]

def create_data(data_dir, use_filename_label=False):
    """Loads images and labels from a directory structure."""
    data = []
    all_labels = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    for label in tqdm(all_labels, desc="Loading Data (Folders)"):
        label_path = os.path.join(data_dir, label)
        
        if not use_filename_label and label not in label_to_index:
            continue  # Skip unknown labels

        images = sorted(os.listdir(label_path))[:MAX_IMAGES_PER_CLASS]
        img_paths = [os.path.join(label_path, img_name) for img_name in images if os.path.isfile(os.path.join(label_path, img_name))]
        label_indices = [label_to_index[label]] * len(img_paths)

        for img_path, label_idx in zip(img_paths, label_indices):
            processed = process_image(img_path, label_idx)
            if processed:
                data.append(processed)

    shuffle(data)
    return data

def create_data_filename(data_dir, start=0, end=1000):
    """Loads images from a directory where labels are inferred from filenames."""
    data = []
    images = sorted(os.listdir(data_dir))[start:end]

    for img_name in tqdm(images, desc="Loading Data (Filenames)"):
        img_path = os.path.join(data_dir, img_name)

        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0

        label = img_name[0].upper()
        if label in label_to_index:
            label_index = label_to_index[label]
            data.append([img, label_index])

    shuffle(data)
    return data

# Load Training Data
train_data = create_data(TRAIN_DIR)
shuffle(train_data)

# ========================
# DATA PREPARATION
# ========================
train = train_data[:round(-0.1*len(train_data))]
val = train_data[round(-0.1*len(train_data)):]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_train = np.array([i[1] for i in train])

X_val = np.array([i[0] for i in val]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_val = np.array([i[1] for i in val])

# ========================
# CNN MODEL DESIGN
# ========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    GlobalAveragePooling2D(),

    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

# ========================
# COMPILATION
# ========================
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ========================
# TRAINING (Save Best Model)
# ========================
checkpoint = ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,  # Increased since no augmentation
    batch_size=32,
    callbacks=[checkpoint]
)

# ========================
# PLOTTING RESULTS
# ========================
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.show()
