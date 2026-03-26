import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------------
# Load Model
# -------------------------------
model = load_model("model.h5")   # apna trained model yaha rakho

# -------------------------------
# App Title
# -------------------------------
st.title("🐶 Cat vs Dog Classifier")
st.write("Upload an image and the model will predict whether it's a Cat or Dog.")

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# -------------------------------
# Prediction Function
# -------------------------------
def predict_image(image):
    img = load_img(image, target_size=(150, 150))   # training size same hona chahiye
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "🐶 Dog"
    else:
        return "🐱 Cat"

# -------------------------------
# Show Result
# -------------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    result = predict_image(uploaded_file)

    st.success(f"Prediction: {result}")
