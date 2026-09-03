import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Set seeds
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)

DATA_DIR = os.path.join("PRAICP-1000-IndiSignLang", "Data")

classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

class_to_index = {c: i for i, c in enumerate(classes)}

image_paths = []
image_labels = []

for c in classes:
    c_path = os.path.join(DATA_DIR, c)
    for f in os.listdir(c_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(c_path, f))
            image_labels.append(c)

df = pd.DataFrame({"image_path": image_paths, "label": image_labels})

# Remove duplicate Data/I/002.jpg
df_clean = df[~df["image_path"].str.replace("\\", "/").str.endswith("Data/I/002.jpg")].reset_index(drop=True)

# Stratified split 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(
    df_clean, test_size=0.20, random_state=42, stratify=df_clean["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_img(path):
    with Image.open(path) as img:
        img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
        return np.array(img, dtype=np.float32) / 255.0

print("Loading images into RAM...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as executor:
    X_train = np.array(list(executor.map(load_img, train_df["image_path"].values)), dtype=np.float32)
    X_val = np.array(list(executor.map(load_img, val_df["image_path"].values)), dtype=np.float32)
    X_test = np.array(list(executor.map(load_img, test_df["image_path"].values)), dtype=np.float32)

y_train = train_df["label"].map(class_to_index).values
y_val = val_df["label"].map(class_to_index).values
y_test = test_df["label"].map(class_to_index).values

print(f"Data loaded in {time.time() - t0:.2f}s!")

# Pre-apply augmentation once for fast training
aug_layer = tf.keras.Sequential([
    layers.RandomRotation(0.05, seed=42),
    layers.RandomZoom(0.10, seed=42),
    layers.RandomTranslation(0.10, 0.10, seed=42)
])

print("Pre-augmenting dataset offline...")
t_aug = time.time()
X_train_aug = aug_layer(X_train, training=True).numpy()
print(f"Augmentation pre-computed in {time.time() - t_aug:.2f}s!")

# CNN Architecture
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(24, activation="softmax")
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(classes)),
    y=y_train
)
class_weights = dict(enumerate(class_weight_values))

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_isl_cnn.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    )
]

print("Starting Fast Stable training...")
t_start = time.time()
history = model.fit(
    X_train_aug, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print(f"Fast Stable training finished in {time.time() - t_start:.2f} seconds!")

model.save("ISL_CNN_Final.keras")
os.makedirs("PRAICP-1000-ISL-Recognition", exist_ok=True)
model.save(os.path.join("PRAICP-1000-ISL-Recognition", "ISL_CNN_Final.keras"))
print("Saved ISL_CNN_Final.keras successfully in both root and PRAICP-1000-ISL-Recognition!")

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=64)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")
