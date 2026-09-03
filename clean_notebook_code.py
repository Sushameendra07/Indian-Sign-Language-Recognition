

# ========================================# Cell 3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Libraries imported successfully!")
print("TensorFlow Version:", tf.__version__)

# Cell 6
import os
import subprocess

zip_path = "/content/PRAICP-1000-IndiSignLang.zip"
extract_path = "/content/ISL_Dataset"

os.makedirs(extract_path, exist_ok=True)

# Check whether 7-Zip is available
check = subprocess.run(
    ["7z", "--help"],
    capture_output=True,
    text=True
)

print("7-Zip available:", check.returncode == 0)

if check.returncode == 0:
    result = subprocess.run(
        ["7z", "x", zip_path, f"-o{extract_path}", "-y"],
        capture_output=True,
        text=True
    )

    print(result.stdout[-5000:])

    if result.returncode == 0:
        print("\nDataset extraction completed.")
    else:
        print("\n7-Zip extraction returned an error.")
        print(result.stderr)
else:
    print("7-Zip is not installed in this environment.")

# Cell 7
# Check the extracted dataset structure

print("Extracted Dataset Structure:\n")

for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, "").count(os.sep)
    indent = "    " * level

    print(f"{indent}{os.path.basename(root)}/")

    # Display only a few files from each directory
    for file in files[:5]:
        print(f"{indent}    {file}")

print("\nTotal files extracted:", sum(len(files) for _, _, files in os.walk(extract_path)))

# Cell 10
# Define the main dataset directory
DATA_DIR = os.path.join(extract_path, "Data")

# Identify class directories
classes = sorted([
    folder
    for folder in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, folder))
])

print("Dataset Directory:", DATA_DIR)
print("Number of Classes:", len(classes))
print("Classes:", classes)

# Cell 13
# Count images available in each class

class_counts = {}

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)

    image_files = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    class_counts[class_name] = len(image_files)

print("Class-wise Image Count:\n")

for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

print("\nTotal Number of Images:", sum(class_counts.values()))

# Cell 16
# Create a DataFrame from the class-wise image counts
class_distribution = pd.DataFrame(
    list(class_counts.items()),
    columns=["Class", "Image Count"]
)

# Plot class distribution
plt.figure(figsize=(14, 6))

sns.barplot(
    data=class_distribution,
    x="Class",
    y="Image Count"
)

plt.title("Class-wise Image Distribution")
plt.xlabel("Sign Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Cell 19
from collections import Counter

# Collect image file extensions
image_extensions = []

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            extension = os.path.splitext(file)[1].lower()
            image_extensions.append(extension)

# Count each image format
extension_counts = Counter(image_extensions)

print("Image Format Distribution:\n")

for extension, count in sorted(extension_counts.items()):
    print(f"{extension}: {count} images")

print("\nTotal Images Analysed:", len(image_extensions))

# Cell 22
from collections import Counter

# Collect image dimensions
image_dimensions = []

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(class_path, file)

            try:
                with Image.open(image_path) as img:
                    image_dimensions.append(img.size)
            except Exception:
                pass

# Count each dimension
dimension_counts = Counter(image_dimensions)

print("Image Dimension Distribution:\n")

for dimension, count in dimension_counts.items():
    print(f"{dimension}: {count} images")

print("\nNumber of images analysed:", len(image_dimensions))
print("Number of unique dimensions:", len(dimension_counts))

# Cell 25
# Analyse image color modes

image_modes = []

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(class_path, file)

            try:
                with Image.open(image_path) as img:
                    image_modes.append(img.mode)
            except Exception:
                pass

# Count each image mode
mode_counts = Counter(image_modes)

print("Image Color Mode Distribution:\n")

for mode, count in mode_counts.items():
    print(f"{mode}: {count} images")

print("\nTotal Images Analysed:", len(image_modes))

# Cell 28
# Identify duplicate images using MD5 hashing

image_hashes = {}
duplicate_images = []

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(class_path, file)

            with open(image_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in image_hashes:
                duplicate_images.append(
                    (image_path, image_hashes[file_hash])
                )
            else:
                image_hashes[file_hash] = image_path

print("Total images checked:", len(image_hashes) + len(duplicate_images))
print("Number of duplicate images:", len(duplicate_images))

if duplicate_images:
    print("\nDuplicate image files:")
    for duplicate, original in duplicate_images:
        print("Duplicate:", duplicate)
        print("Original :", original)

# Cell 31
# Check for invalid or corrupted images

invalid_images = []
total_checked = 0

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(class_path, file)
            total_checked += 1

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except Exception:
                invalid_images.append(image_path)

print("Total images checked:", total_checked)
print("Number of invalid/corrupted images:", len(invalid_images))

if invalid_images:
    print("\nInvalid/Corrupted image files:")
    for image_path in invalid_images:
        print(image_path)

# Cell 34
# Visualize one sample image from each class

fig, axes = plt.subplots(4, 6, figsize=(16, 12))
axes = axes.flatten()

for index, class_name in enumerate(classes):

    class_path = os.path.join(DATA_DIR, class_name)

    image_files = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Select the first available image
    sample_image_path = os.path.join(class_path, image_files[0])

    # Open image for visualization
    with Image.open(sample_image_path) as img:
        axes[index].imshow(img.convert("RGB"))

    axes[index].set_title(f"Class: {class_name}")
    axes[index].axis("off")

plt.suptitle("Sample Image from Each Indian Sign Language Class", fontsize=16)
plt.tight_layout()
plt.show()

# Cell 37
# Select one sample image for pixel value analysis

sample_class = classes[0]
sample_class_path = os.path.join(DATA_DIR, sample_class)

sample_image_file = [
    file for file in os.listdir(sample_class_path)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
][0]

sample_image_path = os.path.join(sample_class_path, sample_image_file)

# Open the sample image and convert it into a NumPy array
with Image.open(sample_image_path) as img:
    sample_image = img.convert("RGB")
    image_array = np.array(sample_image)

print("Sample Class:", sample_class)
print("Sample Image:", sample_image_file)
print("Image Shape:", image_array.shape)
print("Minimum Pixel Value:", image_array.min())
print("Maximum Pixel Value:", image_array.max())
print("Data Type:", image_array.dtype)

# Cell 40
# Create a consolidated dataset summary

print("========== DATASET SUMMARY ==========")
print("Dataset Path:", DATA_DIR)
print("Total Number of Images:", len(image_extensions))
print("Number of Classes:", len(classes))
print("Image Format: .jpg")
print("Image Mode: RGB")
print("Unique Image Dimensions:", len(dimension_counts))
print("Duplicate Images:", len(duplicate_images))
print("Invalid/Corrupted Images:", len(invalid_images))
print("=====================================")

# Cell 44
# Define image preprocessing parameters

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

print("Image Height:", IMG_HEIGHT)
print("Image Width:", IMG_WIDTH)
print("Batch Size:", BATCH_SIZE)

# Cell 47
# Create image file paths and corresponding class labels

image_paths = []
image_labels = []

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(class_path, file))
            image_labels.append(class_name)

# Create DataFrame
image_df = pd.DataFrame({
    "image_path": image_paths,
    "label": image_labels
})

print("Total Images:", len(image_df))
print("Total Classes:", image_df["label"].nunique())

print("\nFirst 5 Records:")
display(image_df.head())

# Cell 50
# First split: 80% training and 20% temporary data
train_df, temp_df = train_test_split(
    image_df,
    test_size=0.20,
    random_state=42,
    stratify=image_df["label"]
)

# Second split: divide the temporary data equally
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# Reset DataFrame indexes
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Training Images:", len(train_df))
print("Validation Images:", len(val_df))
print("Testing Images:", len(test_df))

print("\nTotal Images:", len(train_df) + len(val_df) + len(test_df))

# Cell 53
# Calculate class distribution for each dataset split

train_class_counts = train_df["label"].value_counts().reindex(classes, fill_value=0)
val_class_counts = val_df["label"].value_counts().reindex(classes, fill_value=0)
test_class_counts = test_df["label"].value_counts().reindex(classes, fill_value=0)

# Create comparison DataFrame
split_distribution = pd.DataFrame({
    "Class": classes,
    "Training": train_class_counts.values,
    "Validation": val_class_counts.values,
    "Testing": test_class_counts.values
})

display(split_distribution)

print("Training Images:", split_distribution["Training"].sum())
print("Validation Images:", split_distribution["Validation"].sum())
print("Testing Images:", split_distribution["Testing"].sum())

# Cell 56
# Path of the identified duplicate image
duplicate_path = "/content/ISL_Dataset/Data/I/002.jpg"

# Remove the duplicate image
image_df_clean = image_df[
    image_df["image_path"] != duplicate_path
].reset_index(drop=True)

print("Original Images:", len(image_df))
print("Images After Duplicate Removal:", len(image_df_clean))
print("Images Removed:", len(image_df) - len(image_df_clean))

# Cell 59
train_df, temp_df = train_test_split(
    image_df_clean,
    test_size=0.20,
    random_state=42,
    stratify=image_df_clean["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Training Images:", len(train_df))
print("Validation Images:", len(val_df))
print("Testing Images:", len(test_df))
print("Total Images:", len(train_df) + len(val_df) + len(test_df))

# Cell 62
train_class_counts = (
    train_df["label"]
    .value_counts()
    .reindex(classes, fill_value=0)
)

val_class_counts = (
    val_df["label"]
    .value_counts()
    .reindex(classes, fill_value=0)
)

test_class_counts = (
    test_df["label"]
    .value_counts()
    .reindex(classes, fill_value=0)
)

final_split_distribution = pd.DataFrame({
    "Class": classes,
    "Training": train_class_counts.values,
    "Validation": val_class_counts.values,
    "Testing": test_class_counts.values
})

display(final_split_distribution)

print("Training Images:", final_split_distribution["Training"].sum())
print("Validation Images:", final_split_distribution["Validation"].sum())
print("Testing Images:", final_split_distribution["Testing"].sum())

# Cell 65
class_to_index = {
    class_name: index
    for index, class_name in enumerate(classes)
}

index_to_class = {
    index: class_name
    for class_name, index in class_to_index.items()
}

print("Class-to-Index Mapping:\n")

for class_name, index in class_to_index.items():
    print(f"{class_name} -> {index}")

print("\nTotal Classes:", len(class_to_index))

# Cell 68
def load_and_preprocess_image(path, label):
    # Read image file
    image = tf.io.read_file(path)

    # Decode JPEG image as RGB
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize image to fixed dimensions
    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH]
    )

    # Convert pixel values to float and normalize to 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# Convert class labels into numerical indices
train_labels = train_df["label"].map(class_to_index).values
val_labels = val_df["label"].map(class_to_index).values
test_labels = test_df["label"].map(class_to_index).values

# Get image paths
train_paths = train_df["image_path"].values
val_paths = val_df["image_path"].values
test_paths = test_df["image_path"].values


# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)


# Apply preprocessing
train_dataset = train_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# Shuffle only the training dataset
train_dataset = train_dataset.shuffle(
    buffer_size=len(train_df),
    seed=42
)


# Create batches
train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# Prefetch batches for efficient training
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("TensorFlow datasets created successfully!")

# Cell 71
for images, labels in train_dataset.take(1):
    print("Image Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)
    print("Minimum Pixel Value:", tf.reduce_min(images).numpy())
    print("Maximum Pixel Value:", tf.reduce_max(images).numpy())

# Cell 74
train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
test_batches = tf.data.experimental.cardinality(test_dataset).numpy()

print("Training Batches:", train_batches)
print("Validation Batches:", val_batches)
print("Testing Batches:", test_batches)

print("\nBatch Size:", BATCH_SIZE)

# Cell 77
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10)
], name="data_augmentation")

print("Data augmentation pipeline created successfully!")
print("\nAugmentation Layers:")
for layer in data_augmentation.layers:
    print("-", layer.name)

# Cell 80
# Take one batch from the training dataset
sample_images, sample_labels = next(iter(train_dataset))

# Select the first image from the batch
original_image = sample_images[0]
original_label = sample_labels[0].numpy()

# Create augmented versions
augmented_images = [
    data_augmentation(
        tf.expand_dims(original_image, axis=0),
        training=True
    )[0].numpy()
    for _ in range(4)
]

# Display original and augmented images
plt.figure(figsize=(15, 3))

plt.subplot(1, 5, 1)
plt.imshow(original_image.numpy())
plt.title(f"Original\nClass: {index_to_class[original_label]}")
plt.axis("off")

for i, augmented_image in enumerate(augmented_images, start=2):
    plt.subplot(1, 5, i)
    plt.imshow(np.clip(augmented_image, 0, 1))
    plt.title(f"Augmented {i-1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Cell 83
print("===== FINAL PREPROCESSING SUMMARY =====")
print("Total Clean Images:", len(image_df_clean))
print("Number of Classes:", len(classes))
print("Training Images:", len(train_df))
print("Validation Images:", len(val_df))
print("Testing Images:", len(test_df))
print("Image Size:", f"{IMG_HEIGHT} x {IMG_WIDTH}")
print("Batch Size:", BATCH_SIZE)
print("Training Batches:", train_batches)
print("Validation Batches:", val_batches)
print("Testing Batches:", test_batches)
print("Normalization Range: 0 to 1")
print("Data Augmentation: Enabled")
print("=======================================")

# Cell 87
model = models.Sequential([

    # Input layer
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Data augmentation
    data_augmentation,

    # Convolution Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 4
    layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Feature aggregation
    layers.GlobalAveragePooling2D(),

    # Fully connected layer
    layers.Dense(128, activation="relu"),

    # Dropout for regularization
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(len(classes), activation="softmax")
])

print("CNN model created successfully!")

# Cell 90
model.summary()

# Cell 93
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

print("Model compiled successfully!")
print("Optimizer: Adam")
print("Learning Rate: 0.001")
print("Loss Function: Sparse Categorical Crossentropy")
print("Metric: Accuracy")

# Cell 96
# Calculate class weights using only the training labels
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(classes)),
    y=train_labels
)

class_weights = {
    class_index: weight
    for class_index, weight in enumerate(class_weight_values)
}

class_weight_df = pd.DataFrame({
    "Class": classes,
    "Class Index": np.arange(len(classes)),
    "Training Images": train_class_counts.values,
    "Class Weight": class_weight_values
})

display(class_weight_df)

print("Number of Class Weights:", len(class_weights))
print("Minimum Class Weight:", class_weight_values.min())
print("Maximum Class Weight:", class_weight_values.max())

# Cell 99
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_isl_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
]

print("Training callbacks configured successfully!")
print("EarlyStopping: Enabled")
print("ReduceLROnPlateau: Enabled")
print("ModelCheckpoint: Enabled")

# Cell 102
EPOCHS = 20

print("Starting CNN training...")
print("Maximum Epochs:", EPOCHS)
print("Training Batches per Epoch:", train_batches)
print("Validation Batches per Epoch:", val_batches)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("\nCNN training completed!")

# Cell 105
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Cell 106
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Cell 109
test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\n===== TEST SET EVALUATION =====")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)
print("===============================")

# Cell 112
# Generate predictions for the complete test dataset
y_prob = model.predict(
    test_dataset,
    verbose=1
)

# Convert probabilities to predicted class indices
y_pred = np.argmax(y_prob, axis=1)

# Actual labels
y_true = test_labels

# Convert numerical indices back to class names
y_pred_labels = [
    index_to_class[index]
    for index in y_pred
]

y_true_labels = [
    index_to_class[index]
    for index in y_true
]

print("\n===== PREDICTION SUMMARY =====")
print("Number of Test Samples:", len(y_true))
print("Number of Predictions:", len(y_pred))
print("Number of Classes:", len(classes))
print("==============================")

# Cell 115
report = classification_report(
    y_true_labels,
    y_pred_labels,
    labels=classes,
    target_names=classes,
    digits=4
)

print("===== CLASSIFICATION REPORT =====")
print(report)
print("=================================")

# Cell 118
cm = confusion_matrix(
    y_true_labels,
    y_pred_labels,
    labels=classes
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.title("Confusion Matrix - Indian Sign Language CNN")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# Cell 121
  FINAL_MODEL_PATH = "ISL_CNN_Final.keras"

model.save(FINAL_MODEL_PATH)

print("Final model saved successfully!")
print("Model Path:", FINAL_MODEL_PATH)
print("Best Validation Accuracy:", max(history.history["val_accuracy"]))
print("Test Accuracy:", test_accuracy)

# Cell 124
loaded_model = tf.keras.models.load_model("ISL_CNN_Final.keras")

print("Saved model loaded successfully!")
print("Loaded Model Input Shape:", loaded_model.input_shape)
print("Loaded Model Output Shape:", loaded_model.output_shape)

# Cell 127
# Take one batch from the test dataset
sample_test_images, sample_test_labels = next(iter(test_dataset))

# Generate predictions using the loaded model
sample_predictions = loaded_model.predict(
    sample_test_images,
    verbose=0
)

# Select the first test image
sample_index = 0

# Get predicted class and confidence
predicted_index = np.argmax(sample_predictions[sample_index])
predicted_class = index_to_class[predicted_index]
prediction_confidence = sample_predictions[sample_index][predicted_index]

# Get actual class
actual_index = sample_test_labels[sample_index].numpy()
actual_class = index_to_class[actual_index]

print("===== SAMPLE PREDICTION =====")
print("Actual Class:", actual_class)
print("Predicted Class:", predicted_class)
print("Prediction Confidence:", prediction_confidence)
print("Prediction Confidence (%):", prediction_confidence * 100)
print("=============================")

# Cell 130
def predict_single_image(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert to NumPy array
    image_array = np.array(image_resized, dtype=np.float32)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)

    # Generate prediction
    prediction = loaded_model.predict(
        image_batch,
        verbose=0
    )

    # Get predicted class
    predicted_index = np.argmax(prediction[0])
    predicted_class = index_to_class[predicted_index]

    # Get confidence
    confidence = prediction[0][predicted_index]

    # Display image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(
        f"Predicted Sign: {predicted_class}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )
    plt.axis("off")
    plt.show()

    print("===== SINGLE IMAGE PREDICTION =====")
    print("Image:", image_path)
    print("Predicted Sign:", predicted_class)
    print("Confidence:", f"{confidence * 100:.2f}%")
    print("===================================")

    return predicted_class, confidence


print("Single-image prediction function created successfully!")

# Cell 133
# Select one image from the test dataset
sample_image_path = test_df.iloc[0]["image_path"]
actual_class = test_df.iloc[0]["label"]

print("Actual Class:", actual_class)
print("Image Path:", sample_image_path)

# Predict the sign
predicted_class, confidence = predict_single_image(
    sample_image_path
)

# Cell 136
final_results = pd.DataFrame({
    "Metric": [
        "Clean Dataset Images",
        "Number of Classes",
        "Training Images",
        "Validation Images",
        "Testing Images",
        "Image Size",
        "Batch Size",
        "CNN Parameters",
        "Best Validation Accuracy",
        "Test Accuracy",
        "Test Loss"
    ],
    "Value": [
        len(image_df_clean),
        len(classes),
        len(train_df),
        len(val_df),
        len(test_df),
        f"{IMG_HEIGHT} x {IMG_WIDTH}",
        BATCH_SIZE,
        model.count_params(),
        f"{max(history.history['val_accuracy']) * 100:.2f}%",
        f"{test_accuracy * 100:.2f}%",
        f"{test_loss:.4f}"
    ]
})

display(final_results)

# Cell 140
# Select 12 random images from the test dataset
sample_test_df = test_df.sample(
    n=12,
    random_state=42
).reset_index(drop=True)

prediction_results = []

for _, row in sample_test_df.iterrows():

    image_path = row["image_path"]
    actual_class = row["label"]

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))

    image_array = np.array(
        image_resized,
        dtype=np.float32
    ) / 255.0

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict
    prediction = loaded_model.predict(
        image_batch,
        verbose=0
    )

    predicted_index = np.argmax(prediction[0])
    predicted_class = index_to_class[predicted_index]
    confidence = prediction[0][predicted_index]

    prediction_results.append({
        "Image": os.path.basename(image_path),
        "Actual": actual_class,
        "Predicted": predicted_class,
        "Confidence (%)": round(confidence * 100, 2),
        "Correct": actual_class == predicted_class
    })

prediction_results_df = pd.DataFrame(prediction_results)

display(prediction_results_df)

print("\nCorrect Predictions:",
      prediction_results_df["Correct"].sum())

print("Incorrect Predictions:",
      (~prediction_results_df["Correct"]).sum())

print("Sample Accuracy:",
      f"{prediction_results_df['Correct'].mean() * 100:.2f}%")

# Cell 143
plt.figure(figsize=(16, 12))

for i, row in sample_test_df.iterrows():

    image_path = row["image_path"]
    actual_class = row["label"]

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(
        image_resized,
        dtype=np.float32
    ) / 255.0

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    prediction = loaded_model.predict(
        image_batch,
        verbose=0
    )

    predicted_index = np.argmax(prediction[0])
    predicted_class = index_to_class[predicted_index]
    confidence = prediction[0][predicted_index]

    # Display
    plt.subplot(3, 4, i + 1)
    plt.imshow(image)

    if actual_class == predicted_class:
        title = (
            f"Actual: {actual_class}\n"
            f"Predicted: {predicted_class}\n"
            f"Confidence: {confidence * 100:.2f}%"
        )
    else:
        title = (
            f"Actual: {actual_class}\n"
            f"Predicted: {predicted_class}\n"
            f"Confidence: {confidence * 100:.2f}%"
        )

    plt.title(title)
    plt.axis("off")

plt.suptitle(
    "Indian Sign Language - Multiple Test Predictions",
    fontsize=16
)

plt.tight_layout()
plt.show()