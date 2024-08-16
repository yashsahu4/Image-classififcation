import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Title and description
st.title("Image Classification with Keras")
st.write("Upload an image to classify it using your trained model.")

# Load your pre-trained model
@st.cache_resource
def load_trained_model():
    model = load_model("baseline (1).keras")  # Ensure this path is correct
    return model

model = load_trained_model()

# Image preprocessing function
def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')  # Convert to grayscale
    image = image.resize((180, 180))  # Resize to match model input shape
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)

    # Display the result using the threshold of 0.65
    if prediction[0] > 0.68:
        st.write("The model predicts: **Class 1**")
    else:
        st.write("The model predicts: **Class 0**")
