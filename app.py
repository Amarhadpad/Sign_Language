from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

from sign_pipeline import load_labels, prepare_hand_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "Model" / "labels.txt"
IMAGE_SIZE = 224
OFFSET = 20

st.set_page_config(page_title="Sign Language Converter", page_icon="🤟")
st.title("Sign Language Converter")
st.write(
    "Show a sign to your camera or upload a photo. The app detects one hand "
    "and predicts among the signs supported by this model."
)


@st.cache_resource
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource
def load_detector() -> HandDetector:
    return HandDetector(maxHands=1)


@st.cache_resource
def detector_lock() -> Lock:
    return Lock()


@st.cache_data
def get_labels() -> tuple[str, ...]:
    return tuple(load_labels(LABELS_PATH))


def classify_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not read that image. Try another image file.")
        return None

    detector = load_detector()
    with detector_lock():
        hands, annotated_frame = detector.findHands(frame.copy())
    if not hands:
        st.warning("No hand detected. Try a clearer photo with one hand in view.")
        return None

    prepared = prepare_hand_image(frame, hands[0]["bbox"], OFFSET, 300)
    if prepared is None:
        st.warning("Could not prepare the detected hand. Try another photo.")
        return None

    _, _, hand_image = prepared
    resized_image = cv2.resize(hand_image, (IMAGE_SIZE, IMAGE_SIZE))
    model_input = (resized_image.astype(np.float32) / 127.0 - 1.0)[np.newaxis, ...]
    scores = load_model()(model_input, training=False).numpy()[0]
    return annotated_frame, scores


input_method = st.radio("Image source", ["Take a photo", "Upload an image"], horizontal=True)
image_input = None
if input_method == "Take a photo":
    camera_image = st.camera_input("Capture a hand sign")
    image_input = camera_image
else:
    image_input = st.file_uploader(
        "Choose a hand-sign photo",
        type=["jpg", "jpeg", "png"],
    )

if image_input is not None:
    result = classify_image(image_input.getvalue())
    if result is not None:
        annotated_frame, scores = result
        labels = get_labels()
        if len(scores) != len(labels):
            st.error(
                f"The model returned {len(scores)} scores, but "
                f"{len(labels)} labels are configured."
            )
        else:
            best_index = int(np.argmax(scores))
            st.subheader(f"Recognized sign: {labels[best_index]}")
            st.metric("Model confidence", f"{float(scores[best_index]):.1%}")
            st.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                caption="Hand detected in the submitted image",
                use_container_width=True,
            )
            st.caption(
                "Predictions are for a single image, not a continuous video stream. "
                "Accuracy depends on the training data and image conditions."
            )
