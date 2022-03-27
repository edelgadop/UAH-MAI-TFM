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
age_predictor = tf.keras.models.load_model("./models/age_predictor")

st.title("Age and gender detection using CNNs")
st.text("This application predicts the age and gender from an input image")

st.header("Introduce an image")
image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

prediction = None
prediction_prob = None
age = None

if image_file is not None:
    img = load_image(image_file)
    grayscale_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)  # grayscale transformation
    st.image(img, width=250)
    test_sample = cv2.resize(grayscale_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC) / 255
    test_sample_tensor = np.expand_dims(np.expand_dims(test_sample, axis=0), axis=3)
    prediction_prob = gender_classifier.predict(test_sample_tensor)[0][0]
    prediction_age = age_predictor.predict(test_sample_tensor)[0][0]
    prediction = np.int32(np.round(prediction_prob, 0))
    age = np.int32(np.round(prediction_age))

st.header("Prediction")

if prediction is not None:
    p = np.round(100*prediction_prob, 2)
    l = prediction
    st.text(f"Gender prediction: {gender_dict[l]} ( Prob. {(100-p)*(1-l)+p*l}%)")
    st.text(f"Age prediction: {age}")
