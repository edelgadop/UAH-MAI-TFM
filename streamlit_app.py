import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np


def load_image(image_file):
    img = Image.open(image_file)
    return img


def get_model_summary(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)


gender_dict = {0: "Female", 1: "Male"}
gender_classifier = tf.keras.models.load_model("./models/gender_classifier")

st.title("Age and gender detection using CNNs")
st.text("This application predicts the age and gender from an input image")

st.header("Model")
st.text(get_model_summary(gender_classifier))

st.header("Introduce an image")
image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

prediction = None

if image_file is not None:
    img = load_image(image_file)
    st.image(img, width=250)
    test_sample = cv2.resize(np.array(img), dsize=(250, 250), interpolation=cv2.INTER_CUBIC) / 255
    test_sample_tensor = np.expand_dims(test_sample, axis=0)
    prediction = np.uint8(gender_classifier.predict(test_sample_tensor))[0][0]

st.header("Prediction")

if prediction is not None:
    st.text(f"Gender prediction: {gender_dict[prediction]}")
