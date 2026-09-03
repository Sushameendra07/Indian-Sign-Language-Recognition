import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "ISL_CNN_Final.keras"
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]
INDEX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully from:", MODEL_PATH)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# Pick a test image from class K
test_dir = os.path.join("PRAICP-1000-IndiSignLang", "Data", "K")
test_file = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0]
test_image_path = os.path.join(test_dir, test_file)

print("Test Image Path:", test_image_path)

# Preprocess image
image = Image.open(test_image_path).convert("RGB")
resized_img = image.resize((224, 224))
img_array = np.array(resized_img, dtype=np.float32) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_batch, verbose=0)[0]
top_idx = int(np.argmax(predictions))
predicted_sign = INDEX_TO_CLASS[top_idx]
confidence_val = float(predictions[top_idx]) * 100.0

print(f"\n===== INFERENCE RESULT =====")
print(f"Predicted Sign: {predicted_sign}")
print(f"Confidence: {confidence_val:.2f}%")

print("\n===== TOP-5 PREDICTIONS =====")
top_5_indices = np.argsort(predictions)[::-1][:5]
for rank, idx in enumerate(top_5_indices, start=1):
    print(f"Rank {rank}: {INDEX_TO_CLASS[idx]} - {predictions[idx]*100.0:.2f}%")
