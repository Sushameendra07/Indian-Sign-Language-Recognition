import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Indian Sign Language Recognition | PRAICP-1000",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS FOR STUNNING MODERN AESTHETICS
# ==========================================
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container Styling */
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2.5rem 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .main-subtitle {
        font-size: 1.15rem;
        color: #c7d2fe;
        font-weight: 400;
        margin-bottom: 0;
    }

    /* Card Styling */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 0.75rem;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.25rem;
    }
    
    /* Metric Card Highlight */
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Prediction Result Banner */
    .prediction-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 50%, #10b981 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);
        margin-bottom: 1.5rem;
    }

    .prediction-sign {
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin: 0.5rem 0;
        color: #ffffff;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .prediction-confidence {
        font-size: 1.5rem;
        font-weight: 600;
        color: #a7f3d0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid #1e293b;
    }

    /* Button Customization */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.65rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%);
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.6);
        transform: translateY(-1px);
    }
    
    /* Flow Step Pills */
    .flow-step {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #6366f1;
        padding: 0.85rem 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & EXACT CLASS MAPPING
# ==========================================
MODEL_PATH = "ISL_CNN_Final.keras"
IMG_HEIGHT = 224
IMG_WIDTH = 224

# EXACT Class Mapping as required by Project Specification:
# A -> 0, B -> 1, ..., Y -> 23 (No 'J')
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]
INDEX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# ==========================================
# CACHED MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_isl_model():
    """Loads the pre-trained CNN model into memory once and caches it."""
    if not os.path.exists(MODEL_PATH):
        # Fallback check if path is in parent directory
        if os.path.exists(os.path.join("..", MODEL_PATH)):
            return tf.keras.models.load_model(os.path.join("..", MODEL_PATH))
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

# Load model quietly
model = load_isl_model()

# ==========================================
# SIDEBAR CONTENT
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hand.png", width=64)
    st.title("Project Metadata")
    st.markdown("---")
    
    st.markdown("### 📋 **Reference**")
    st.info("**PRAICP-1000**")
    
    st.markdown("### 🤟 **Project Name**")
    st.markdown("**Indian Sign Language Recognition**")
    
    st.markdown("### 🎯 **Domain**")
    st.markdown("Computer Vision / AI")
    
    st.markdown("### 🧠 **Model Architecture**")
    st.markdown("Convolutional Neural Network (CNN)")
    
    st.markdown("### 📈 **Performance**")
    st.markdown("- **Test Accuracy:** `88.96%`")
    st.markdown("- **Best Val Accuracy:** `89.74%`")
    st.markdown("- **Total Classes:** `24` *(No 'J')*")
    st.markdown("- **Parameters:** `424,408`")

    st.markdown("---")
    st.markdown("💡 **Tip:** Upload any standard Indian Sign Language hand sign image (`.jpg`, `.jpeg`, `.png`) to perform instant AI classification.")

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="main-title">
        🤟 Indian Sign Language Recognition
    </div>
    <div class="main-subtitle">
        AI-Powered Image Classification using a 424k Parameter CNN Architecture | Project PRAICP-1000
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN APPLICATION TABS
# ==========================================
tab_inference, tab_home, tab_about, tab_performance, tab_workflow = st.tabs([
    "📸 Inference & Prediction",
    "🏠 Home & Overview",
    "🧠 About the Model",
    "📊 Model Performance",
    "🔄 How It Works"
])

# ==========================================
# TAB 1: INFERENCE & PREDICTION
# ==========================================
with tab_inference:
    st.markdown("### 📤 Upload Hand Sign Image")
    st.markdown("Upload a static hand sign image from your local system for real-time classification.")
    
    col_upload, col_display = st.columns([1, 1], gap="large")
    
    uploaded_file = None
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        predict_button = st.button("🔮 Predict Sign", use_container_width=True)

    input_image = None
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error("⚠️ Invalid or corrupted image file. Please upload a valid image.")
            input_image = None
            
    with col_display:
        if input_image is not None:
            st.markdown("#### 🖼️ **Uploaded Image**")
            st.image(input_image, caption="Original Input Image (Displayed without distortion)", use_container_width=True)
        else:
            st.info("👆 Upload an image using the panel on the left to start prediction.")

    st.markdown("---")

    # Perform Prediction when requested or when image is present
    if predict_button or (uploaded_file is not None and st.session_state.get("auto_run", True)):
        if input_image is me = None if 'input_image' not in locals() else input_image:
            pass

        if input_image is None:
            st.warning("⚠️ Please upload an image first before attempting prediction.")
        elif model is None:
            st.error("❌ Model file (`ISL_CNN_Final.keras`) could not be loaded. Please ensure the model file is in the project directory.")
        else:
            try:
                with st.spinner("Processing image through CNN pipeline..."):
                    # ------------------------------------------
                    # DETERMINISTIC PREPROCESSING (EXACT WORKFLOW)
                    # ------------------------------------------
                    # 1. Image converted to RGB (Done during Image.open)
                    # 2. Resize to 224 x 224
                    resized_img = input_image.resize((IMG_WIDTH, IMG_HEIGHT))
                    # 3. Convert to NumPy array
                    img_array = np.array(resized_img, dtype=np.float32)
                    # 4. Normalize pixel values (0 to 1)
                    img_array = img_array / 255.0
                    # 5. Add batch dimension -> (1, 224, 224, 3)
                    img_batch = np.expand_dims(img_array, axis=0)

                    # 6. Pass through loaded CNN
                    predictions = model.predict(img_batch, verbose=0)[0]

                    # 7. Get 24 probability values & find argmax
                    top_idx = int(np.argmax(predictions))
                    predicted_sign = INDEX_TO_CLASS[top_idx]
                    confidence_val = float(predictions[top_idx]) * 100.0

                    # Sort top-5 probabilities
                    top_5_indices = np.argsort(predictions)[::-1][:5]
                    top_5_data = [
                        {
                            "Rank": rank + 1,
                            "Sign": INDEX_TO_CLASS[idx],
                            "Probability": f"{predictions[idx] * 100.0:.2f}%",
                            "RawProb": predictions[idx] * 100.0
                        }
                        for rank, idx in enumerate(top_5_indices)
                    ]

                # ------------------------------------------
                # DISPLAY PREDICTION RESULTS
                # ------------------------------------------
                col_res1, col_res2 = st.columns([1, 1], gap="large")

                with col_res1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9;">
                            Predicted Sign Class
                        </div>
                        <div class="prediction-sign">
                            Sign: {predicted_sign}
                        </div>
                        <div class="prediction-confidence">
                            Confidence: {confidence_val:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### 🎯 **Confidence Indicator**")
                    st.progress(min(confidence_val / 100.0, 1.0))
                    
                    st.metric(
                        label="Prediction Confidence Level",
                        value=f"{confidence_val:.2f}%",
                        delta="High Certainty" if confidence_val > 75 else "Moderate / Check Top 5"
                    )

                with col_res2:
                    st.markdown("#### 🏆 **Top-5 Predicted Classes**")
                    top5_df = pd.DataFrame(top_5_data)[["Rank", "Sign", "Probability"]]
                    st.table(top5_df)

                    st.markdown("#### 📊 **Top-5 Probability Distribution**")
                    plot_df = pd.DataFrame(top_5_data)
                    fig = px.bar(
                        plot_df,
                        x="RawProb",
                        y="Sign",
                        orientation="h",
                        text="Probability",
                        labels={"RawProb": "Probability (%)", "Sign": "Sign Class"},
                        color="RawProb",
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(
                        yaxis=dict(autorange="reversed"),
                        height=240,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#f8fafc")
                    )
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Prediction Error: {e}")

# ==========================================
# TAB 2: HOME & OVERVIEW
# ==========================================
with tab_home:
    st.markdown("### 🤟 Indian Sign Language Recognition")
    st.write(
        "This web application utilizes a custom **Convolutional Neural Network (CNN)** trained to recognize "
        "**24 Indian Sign Language (ISL)** hand-sign classes directly from RGB images. It provides deterministic, "
        "real-time inference with full confidence breakdown across top predictions."
    )
    
    st.markdown("#### 📌 **Project Quick Summary**")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Project Code</div>
            <div class="metric-value" style="color: #818cf8;">PRAICP-1000</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Cleaned Dataset</div>
            <div class="metric-value" style="color: #38bdf8;">4,971</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Classes</div>
            <div class="metric-value" style="color: #34d399;">24</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value" style="color: #f43f5e;">88.96%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_h1, col_h2 = st.columns(2, gap="large")
    with col_h1:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #818cf8; margin-bottom: 0.75rem;">📂 Dataset Information</h4>
            <ul>
                <li><b>Original Images:</b> 4,972 images</li>
                <li><b>Duplicate Detection:</b> 1 exact duplicate removed (<code>Data/I/002.jpg</code>)</li>
                <li><b>Final Clean Dataset:</b> 4,971 images</li>
                <li><b>Training Split (80%):</b> 3,976 images</li>
                <li><b>Validation Split (10%):</b> 497 images</li>
                <li><b>Testing Split (10%):</b> 498 images</li>
                <li><b>Class Count:</b> 24 classes (No 'J' class)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_h2:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #38bdf8; margin-bottom: 0.75rem;">🔤 Recognized Classes (24 Total)</h4>
            <p>The 24 alphabet signs supported by this model are:</p>
            <div style="font-size: 1.15rem; font-weight: 700; color: #a7f3d0; word-spacing: 0.4rem;">
                A B C D E F G H I K L M N O P Q R S T U V W X Y
            </div>
            <br>
            <p><i>Note: The character 'J' is excluded from this dataset as it involves dynamic continuous movement.</i></p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: ABOUT THE MODEL
# ==========================================
with tab_about:
    st.markdown("### 🧠 CNN Architecture Details")
    st.write(
        "The model is a custom multi-block Convolutional Neural Network designed specifically for multi-class sign language recognition. "
        "All 424,408 parameters are trainable."
    )
    
    col_a1, col_a2 = st.columns([1, 1], gap="large")
    
    with col_a1:
        st.markdown("#### ⚙️ **Hyperparameter Specs**")
        hyper_df = pd.DataFrame({
            "Hyperparameter / Property": [
                "Model Type",
                "Total Trainable Parameters",
                "Input Dimension",
                "Output Channels",
                "Optimizer",
                "Initial Learning Rate",
                "Loss Function",
                "Training Epochs",
                "Data Augmentation (Training)",
                "Class Weighting"
            ],
            "Value": [
                "Convolutional Neural Network (CNN)",
                "424,408",
                "224 x 224 x 3 (RGB)",
                "24 Softmax Classes",
                "Adam",
                "0.001",
                "Sparse Categorical Crossentropy",
                "20",
                "Rotation (5%), Zoom (10%), Translation (10%)",
                "Balanced Class Weights"
            ]
        })
        st.table(hyper_df)
        
    with col_a2:
        st.markdown("#### 🏗️ **Layer Pipeline Architecture**")
        st.markdown("""
        <div class="flow-step">1️⃣ <b>Input Layer:</b> (None, 224, 224, 3) RGB Tensor</div>
        <div class="flow-step">2️⃣ <b>Augmentation Layer:</b> Rotation(0.05), Zoom(0.1), Translation(0.1)</div>
        <div class="flow-step">3️⃣ <b>Conv Block 1:</b> Conv2D 32 (3x3, ReLU, same) ➔ MaxPooling2D (2x2)</div>
        <div class="flow-step">4️⃣ <b>Conv Block 2:</b> Conv2D 64 (3x3, ReLU, same) ➔ MaxPooling2D (2x2)</div>
        <div class="flow-step">5️⃣ <b>Conv Block 3:</b> Conv2D 128 (3x3, ReLU, same) ➔ MaxPooling2D (2x2)</div>
        <div class="flow-step">6️⃣ <b>Conv Block 4:</b> Conv2D 256 (3x3, ReLU, same) ➔ MaxPooling2D (2x2)</div>
        <div class="flow-step">7️⃣ <b>Feature Aggregation:</b> GlobalAveragePooling2D</div>
        <div class="flow-step">8️⃣ <b>Dense Classifier:</b> Dense(128, ReLU) ➔ Dropout(0.5)</div>
        <div class="flow-step">9️⃣ <b>Output Layer:</b> Dense(24, Softmax)</div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 4: MODEL PERFORMANCE
# ==========================================
with tab_performance:
    st.markdown("### 📊 Actual Model Evaluation & Results")
    st.write("Below are the exact quantitative metrics recorded during training and testing of `ISL_CNN_Final.keras`.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Best Validation Accuracy", "89.74%", "Epoch 19")
    with col_p2:
        st.metric("Final Test Accuracy", "88.96%", "498 Test Samples")
    with col_p3:
        st.metric("Test Loss", "0.2750", "Sparse Categorical CE")

    st.markdown("---")
    
    col_c1, col_c2 = st.columns(2, gap="large")
    
    with col_c1:
        st.markdown("#### 📈 **Classification Report Metrics**")
        report_df = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-Score"],
            "Macro Average": ["89.51%", "87.10%", "86.20%"],
            "Weighted Average": ["90.15%", "88.96%", "88.13%"]
        })
        st.table(report_df)
        
        st.markdown("#### ✅ **Strongest Performing Classes**")
        st.success("Classes with highest prediction accuracy: **A, C, D, G, H, I, K, P, Q**")

    with col_c2:
        st.markdown("#### ⚠️ **Challenging Classes & Confusion Patterns**")
        st.warning("Classes with lower relative precision: **V, W, F, N, Y**")
        
        st.markdown("""
        <div class="custom-card">
            <b>Observed Confusion Patterns during Evaluation:</b>
            <ul>
                <li><code>F ➔ B</code> (Hand shape resemblance)</li>
                <li><code>W ➔ L</code> / <code>W ➔ U</code></li>
                <li><code>N ➔ O</code></li>
                <li><code>V ➔ U</code> / <code>V ➔ W</code></li>
                <li><code>Y ➔ E</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 5: HOW IT WORKS
# ==========================================
with tab_workflow:
    st.markdown("### 🔄 End-to-End Inference Workflow")
    st.write("Step-by-step processing pipeline applied to every user image upload:")

    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 0.75rem; max-width: 700px; margin: 0 auto;">
        <div class="flow-step">1️⃣ <b>User Upload:</b> Image received via Streamlit file uploader (.jpg, .png)</div>
        <div style="text-align: center; color: #818cf8; font-weight: bold;">↓</div>
        <div class="flow-step">2️⃣ <b>RGB Conversion:</b> Image converted to 3-channel RGB format</div>
        <div style="text-align: center; color: #818cf8; font-weight: bold;">↓</div>
        <div class="flow-step">3️⃣ <b>Spatial Resizing:</b> Image resized to exactly 224 × 224 pixels</div>
        <div style="text-align: center; color: #818cf8; font-weight: bold;">↓</div>
        <div class="flow-step">4️⃣ <b>Pixel Normalization:</b> Scale uint8 values [0, 255] to float32 [0.0, 1.0]</div>
        <div style="text-align: center; color: #818cf8; font-weight: bold;">↓</div>
        <div class="flow-step">5️⃣ <b>Tensor Shaping:</b> Add batch dimension to form tensor shape (1, 224, 224, 3)</div>
        <div style="text-align: center; color: #818cf8; font-weight: bold;">↓</div>
        <div class="flow-step">6️⃣ <b>CNN Pass:</b> Feed tensor forward through loaded 424,408 parameter CNN</div>
        <div style="text-align: center; color: #818cf8; font-weight: bold;">↓</div>
        <div class="flow-step">7️⃣ <b>Softmax Output:</b> Receive 24 class probability values (sum to 1.0)</div>
        <div style="text-align: center; color: #818cf8; font-weight: bold;">↓</div>
        <div class="flow-step">8️⃣ <b>Class Mapping:</b> Argmax maps numerical index to exact letter (A-Y) + Confidence %</div>
    </div>
    """, unsafe_allow_html=True)
