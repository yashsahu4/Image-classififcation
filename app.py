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
    if image.mode != 'L':  # Convert to grayscale if needed
        image = image.convert('L')
    image = image.resize((180, 180))  # Resize to match the input shape expected by your model
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
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Display the shape and stats of the processed image for debugging
    st.write(f'Processed image shape: {processed_image.shape}')
    st.write(f'Min value: {np.min(processed_image)}, Max value: {np.max(processed_image)}')

    # Make predictions
    prediction = model.predict(processed_image)

    # Display the raw prediction
    st.write(f'Raw prediction: {prediction}')

    # Display the result
    if prediction[0] > 0.5:
        st.write("The model predicts: **Class 1**")
    else:
        st.write("The model predicts: **Class 0**")
