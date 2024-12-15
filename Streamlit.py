# Streamlit Application
import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Base directory for images
base_dir = r"D:\DR Detection-20241209T040612Z-001\DR Detection\2217041\gaussian_filtered_images\gaussian_filtered_images"

@st.cache_resource
def load_model():
    # Load the updated model (best_model.keras)
    model = tf.keras.models.load_model("best_model.keras")
    return model

model = load_model()

# Streamlit UI
st.title("Diabetic Retinopathy Detection")
st.subheader("Select an Image for Prediction")

# Restrict folder selection to "Moderate" and "No_DR"
allowed_folders = ["Moderate", "No_DR"]
selected_folder = st.selectbox("Choose a folder", allowed_folders)

# Get images from the selected folder
folder_path = os.path.join(base_dir, selected_folder)
images = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
selected_image = st.selectbox("Choose an image", images)

if selected_image:
    image_path = os.path.join(folder_path, selected_image)
    image = Image.open(image_path)
    st.image(image, caption=f"Selected Image: {selected_image}", use_column_width=True)

    # Preprocess the image (match training preprocessing)
    image = image.resize((256, 256))  # Ensure resizing matches training
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]

    # Convert grayscale images to RGB if necessary
    if image_array.shape[-1] != 3:
        image_array = np.stack((image_array,) * 3, axis=-1)

    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(image_array)
    st.write("Raw Predictions:", predictions)  # Debugging: Show raw predictions

    # Map class indices to class names
    class_indices = {0: "Mild", 1: "Moderate", 2: "No_DR", 3: "Proliferate_DR", 4: "Severe"}
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_indices.get(predicted_class_index, "Unknown")

    st.subheader(f"Prediction: {predicted_class}")

    # Healthcare tips
    if predicted_class != "No_DR":
        st.markdown("""
        **Healthcare Tips:**
        - Maintain healthy blood sugar levels.
        - Schedule regular eye check-ups with an ophthalmologist.
        - Follow a balanced diet rich in fruits and vegetables.
        - Exercise regularly to manage diabetes.
        - Avoid smoking and excessive alcohol consumption.
        """)
    else:
        st.markdown("No signs of diabetic retinopathy detected. Continue regular check-ups for preventive care.")
else:
    st.info("Please select an image from the folder for prediction.")
