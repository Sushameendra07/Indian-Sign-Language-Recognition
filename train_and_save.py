import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Set seeds
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

DATA_DIR = os.path.join("PRAICP-1000-IndiSignLang", "Data")

classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

class_to_index = {c: i for i, c in enumerate(classes)}
index_to_class = {i: c for i, c in enumerate(classes)}

image_paths = []
image_labels = []

for c in classes:
    c_path = os.path.join(DATA_DIR, c)
    for f in os.listdir(c_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(c_path, f))
            image_labels.append(c)

df = pd.DataFrame({"image_path": image_paths, "label": image_labels})

# Remove duplicate
df_clean = df[~df["image_path"].str.replace("\\", "/").str.endswith("Data/I/002.jpg")].reset_index(drop=True)
print(f"Cleaned dataset: {len(df_clean)} images across {len(classes)} classes")

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

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_labels = train_df["label"].map(class_to_index).values
val_labels = val_df["label"].map(class_to_index).values
test_labels = test_df["label"].map(class_to_index).values

train_paths = train_df["image_path"].values
val_paths = val_df["image_path"].values
test_paths = test_df["image_path"].values

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_dataset = test_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10)
], name="data_augmentation")

# CNN Architecture
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
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
    y=train_labels
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

print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

model.save("ISL_CNN_Final.keras")
print("Saved ISL_CNN_Final.keras successfully!")

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
