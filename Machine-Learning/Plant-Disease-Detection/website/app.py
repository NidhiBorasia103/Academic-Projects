
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# File locations
PROJECT_DIR = Path(r"C:\PlantDiseaseProject")
MODEL_PATH = PROJECT_DIR / "models" / "plant_disease_mobilenetv2.keras"
MAPPING_PATH = PROJECT_DIR / "results" / "class_mapping.json"

# Page settings
st.set_page_config(
    page_title=" Plant Disease Detection",
    layout="centered"
)

# Cwebsite styling
st.markdown("""
<style>
    .stApp {
        background: #f5f8f4;
    }

    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(125deg, #123d2a, #28734b);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        color: white;
    }

    .hero h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #e0eee3;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-title {
        color: #183d2a;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .result-card {
        background: white;
        border: 1px solid #dce8dc;
        border-radius: 16px;
        padding: 1.4rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }

    .result-label {
        color: #39734b;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-name {
        color: #183d2a;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    .confidence-number {
        color: #28734b;
        font-size: 2.3rem;
        font-weight: 750;
    }

    .small-note {
        color: #607064;
        font-size: 0.85rem;
    }

    div.stButton > button {
        background: #28734b;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        background: #1b5837;
        color: white;
        border: none;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #94b59a;
        border-radius: 14px;
        padding: 1rem;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Load disease names in the correct output order
@st.cache_data
def load_class_names():
    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        class_to_index = json.load(file)

    return [
        name
        for name, index in sorted(
            class_to_index.items(),
            key=lambda item: item[1]
        )
    ]

# Check required files
if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not MAPPING_PATH.is_file():
    st.error(f"Class mapping file not found: {MAPPING_PATH}")
    st.stop()


# Website heading
st.markdown("""
<div class="hero">
    <h1>Plant Disease Detection & Classification</h1>
    <br>
    <p>
        Explore your plant's health with a CNN trained to classify
        38 PlantVillage plant and disease categories.
    </p>
</div>
""", unsafe_allow_html=True)


# Research information
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="PlantVillage categories",
        value="38"
    )

with col2:
    st.metric(
        label="MobileNetV2 test accuracy",
        value="92.57%"
    )

st.caption(
    "Test accuracy measured on the held-out PlantVillage test set. "
    "It is not the confidence score for an uploaded photo."
)

st.markdown(
    '<div class="section-title">🔍 Check a leaf image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a leaf photo",
    type=["jpg", "jpeg", "png"],
    help="Choose a clear photo in JPG, JPEG or PNG format."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Your uploaded leaf",
            width=350
        )

        if st.button("Analyze Leaf"):
            with st.spinner("Analyzing the image..."):

                # Prepare the image 
                resized_image = image.resize((150, 150))

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                ) / 255.0

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                model = load_model()
                class_names = load_class_names()

                # Predict the category
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]
                top_indices = np.argsort(predictions)[-3:][::-1]
                best_index = int(top_indices[0])

                predicted_class = class_names[best_index]
                confidence = float(predictions[best_index]) * 100

            display_name = predicted_class.replace(
                "___", " — "
            ).replace("_", " ")

            st.markdown(
                '<div class="section-title">Prediction result</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted category</div>
                <div class="result-name">{display_name}</div>
                <br>
                <div class="result-label">Prediction confidence</div>
                <div class="confidence-number">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(
                float(predictions[best_index]),
                text="Top prediction score"
            )

            st.caption(
                "This percentage is the model's confidence score. "
                "It does not guarantee that the prediction is correct."
            )

           
            with st.expander("View other possible predictions"):
                for index in top_indices[1:]:
                    label = class_names[int(index)]
                    label = label.replace(
                        "___", " — "
                    ).replace("_", " ")

                    score = float(predictions[index]) * 100

                    st.write(f"**{label}** — {score:.2f}%")
                    st.progress(float(predictions[index]))

            st.info(
                "Research note: this model was trained on PlantVillage "
                "images. Results on phone photos taken in natural "
                "conditions may be less reliable."
            )

    except Exception as error:
        st.error("We could not process this image.")
        st.caption(f"Technical details: {error}")

else:
    st.markdown("""
    <div class="result-card">
        <div class="result-name">Ready to explore?</div>
        <p class="small-note">
            Upload a clear plant leaf photo above, then click
            Analyze Leaf to see the predicted category.
        </p>
    </div>
    """, unsafe_allow_html=True)



st.markdown("---")
st.caption(
    "Student research project · "
    "Deep learning for plant disease classification"
)
