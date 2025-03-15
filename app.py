import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import config

# model acc 0.76

# Load trained model
model = tf.keras.models.load_model("pneumonia_model.h5")
img_size = config.img_size

st.title("🩺 AI-Powered Pneumonia Detector (Accuracy 76%)")
st.write("Upload a chest X-ray image, and the AI will predict if pneumonia is present.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]
    result = "Pneumonia Detected" if prediction > 0.7 else "Normal"

    # Display
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write(f"### Prediction: {result}")
    st.write(f"### Score: {prediction}")
