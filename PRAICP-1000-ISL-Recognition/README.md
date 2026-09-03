# 🤟 Indian Sign Language Recognition (PRAICP-1000)

> **AI-Powered Indian Sign Language Image Classification using Convolutional Neural Networks (CNN)**

---

## 📌 Project Overview
- **Project Reference:** `PRAICP-1000`
- **Domain:** Computer Vision / Artificial Intelligence
- **Goal:** Build an image-based Indian Sign Language (ISL) recognition system that accepts static images of hand signs and predicts the corresponding ISL character class.

This repository contains the complete Streamlit web application along with the exact pre-trained CNN model (`ISL_CNN_Final.keras`) designed for deterministic, real-time sign recognition.

---

## 📊 Dataset & Class Information

- **Original Dataset:** 4,972 images
- **Duplicate Detection:** 1 exact duplicate removed (`Data/I/002.jpg`)
- **Final Clean Dataset:** 4,971 images
- **Number of Classes:** 24
- **Classes:** `A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y`
  > ⚠️ **Note:** There is **NO `J` class** in this dataset due to motion/gesture requirements.

### Dataset Split Ratio (80:10:10 Stratified)
- **Training Set:** 3,976 images (80%)
- **Validation Set:** 497 images (10%)
- **Testing Set:** 498 images (10%)

---

## ⚙️ Image Preprocessing & Input Pipeline

For inference, all images undergo deterministic preprocessing:
1. **RGB Conversion:** Convert input image to 3-channel RGB.
2. **Resizing:** Resize target image to **224 × 224** pixels.
3. **Data Type Conversion:** Convert to `float32` NumPy array.
4. **Pixel Normalization:** Scale pixel intensities to `[0.0, 1.0]` by dividing by `255.0`.
5. **Batch Dimension Expansion:** Expand tensor shape to `(1, 224, 224, 3)`.

---

## 🏗️ CNN Architecture & Parameters

The model features a custom 4-Block Convolutional Neural Network architecture built with TensorFlow / Keras:

```
Input Image (224 x 224 x 3)
    ↓
Data Augmentation (Rotation 0.05, Zoom 0.10, Translation 0.10) [Training Only]
    ↓
Conv2D (32 filters, 3x3 kernel, ReLU, padding='same') → MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3 kernel, ReLU, padding='same') → MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3 kernel, ReLU, padding='same') → MaxPooling2D (2x2)
    ↓
Conv2D (256 filters, 3x3 kernel, ReLU, padding='same') → MaxPooling2D (2x2)
    ↓
GlobalAveragePooling2D
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (24 units, Softmax)
```

- **Total Trainable Parameters:** `424,408`
- **Optimizer:** Adam (Initial Learning Rate = `0.001`)
- **Loss Function:** Sparse Categorical Crossentropy
- **Class Balance Strategy:** Balanced Class Weights computed during training.

---

## 📈 Model Performance & Results

- **Best Validation Accuracy:** `89.74%` (Epoch 19)
- **Best Validation Loss:** `0.3016`
- **Final Test Accuracy:** `88.96%`
- **Test Loss:** `0.2750`

### Classification Metrics Summary
| Metric | Macro Average | Weighted Average |
| :--- | :---: | :---: |
| **Precision** | 89.51% | 90.15% |
| **Recall** | 87.10% | 88.96% |
| **F1-Score** | 86.20% | 88.13% |

### Key Observations & Model Behavior
- **Strong Performing Classes:** `A`, `C`, `D`, `G`, `H`, `I`, `K`, `P`, `Q`
- **Challenging Classes:** `V`, `W`, `F`, `N`, `Y`
- **Observed Confusion Pairs:** `F → B`, `W → L`, `N → O`, `V → U`, `V → W`, `W → U`, `Y → E`

---

## 💻 Streamlit Web Application

The interactive web dashboard provides:
1. **Home / Overview:** Project metadata, domain specs, and class distribution overview.
2. **Image Upload:** Upload `.jpg`, `.jpeg`, or `.png` static hand sign images.
3. **Real-Time Inference:** Deterministic prediction using `@st.cache_resource` for instant model loading.
4. **Prominent Prediction Display:** Highlighted predicted class label and exact confidence percentage (2 decimal places).
5. **Top-5 Predictions & Interactive Bar Chart:** Sorted probability breakdown across top 5 candidate classes.
6. **Model Architecture & System Info:** Comprehensive summary of hyperparameters, layers, and total parameters.
7. **Interactive Workflow Diagram:** Step-by-step pipeline visualization from raw upload to Softmax output.

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Clone / Navigate to Directory
```bash
cd PRAICP-1000-ISL-Recognition
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run app.py
```

---

## ⚠️ Limitations & Future Improvements
- **Static Images Only:** Designed for single hand sign images; does not process continuous video streams or dynamic gesture sequences.
- **Lighting & Background Sensitivity:** Performance is optimal when hand gestures are clearly framed against relatively non-distracting backgrounds.
- **Future Work:** Integrate real-time webcam feed via OpenCV/MediaPipe, expand dataset for dynamic signs (e.g., character 'J'), and explore lightweight architectures (e.g., MobileNetV3) for edge deployment.
