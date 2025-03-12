import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                     BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ========================
# GPU CONFIGURATION
# ========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Prevents TensorFlow from consuming all VRAM
        print("✅ GPU Enabled:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU detected, using CPU instead.")

# ========================
# CONFIGURATION & SETUP
# ========================
IMG_SIZE = 64
BATCH_SIZE = 128  # Adjusted for GPU
EPOCHS = 10
MODEL_NAME = 'asl_cnn_classifier.h5'
MAX_IMAGES_PER_CLASS = 3000
TRAIN_DIR = r"C:\Users\aniru\OneDrive\Documents\ASL-ENGLISH-Translator\asl_alphabet_train\asl_alphabet_train"
APPEND_TRAIN = r"C:\Users\aniru\OneDrive\Documents\ASL-ENGLISH-Translator\train\images"

# Enable mixed precision for GPU acceleration
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Enable XLA for just-in-time compilation (boosts performance)
tf.config.optimizer.set_jit(True)

# ========================
# DATA LOADING FUNCTIONS
# ========================
def load_labels():
    """Load labels from folder names."""
    labels = sorted([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])
    return labels

label_names = load_labels()
label_to_index = {label: idx for idx, label in enumerate(label_names)}
num_classes = len(label_names)

def process_image(img_path, label_index):
    """Load, preprocess, and return image and label index."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return img, label_index

def create_data(data_dir):
    """Loads images with multithreading."""
    data = []
    all_labels = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    for label in tqdm(all_labels, desc="Loading Data"):
        label_path = os.path.join(data_dir, label)
        if label not in label_to_index:
            continue
        
        images = sorted(os.listdir(label_path))[:MAX_IMAGES_PER_CLASS]  
        img_paths = [os.path.join(label_path, img) for img in images if os.path.isfile(os.path.join(label_path, img))]

        with ThreadPoolExecutor() as executor:
            results = executor.map(process_image, img_paths, [label_to_index[label]] * len(img_paths))
        
        data.extend([res for res in results if res])  

    shuffle(data)
    return data

# Load Training Data
train_data = create_data(TRAIN_DIR)
append_train_data = create_data(APPEND_TRAIN)

# Merge datasets and shuffle
train_data.extend(append_train_data)
shuffle(train_data)

# Splitting dataset
train = train_data[:-1000]
val = train_data[-1000:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_train = np.array([i[1] for i in train])

X_val = np.array([i[0] for i in val]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_val = np.array([i[1] for i in val])

# ========================
# TF.DATA PIPELINE (OPTIMIZED FOR GPU)
# ========================
def prepare_dataset(X, Y):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(len(X)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # Prefetch improves GPU utilization
    return dataset

train_dataset = prepare_dataset(X_train, Y_train)
val_dataset = prepare_dataset(X_val, Y_val)

# ========================
# CNN MODEL (OPTIMIZED FOR GPU)
# ========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    GlobalAveragePooling2D(),  # Replaces Flatten() for efficiency
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax', dtype='float32')  # Force FP32 output for precision
])

# ========================
# COMPILATION
# ========================
optimizer = Adam(learning_rate=0.001, jit_compile=True)  # JIT Compilation Enabled
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ========================
# TRAINING (SAVE BEST MODEL)
# ========================
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# ========================
# EVALUATION
# ========================
val_loss, val_acc = model.evaluate(val_dataset)
print(f"Validation Accuracy: {val_acc:.4f}")

# Save Final Model
model.save(MODEL_NAME)
