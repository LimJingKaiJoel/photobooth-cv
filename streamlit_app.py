import streamlit as st
from PIL import Image
import run
import cv2

# Set up the title and description
st.title("Image Background Change and Style Transform")
st.write("Upload an image to change its background and transform its style.")

# File uploader
uploaded_file = st.file_uploader("Choose an input image for background change and style transform.", type=["jpg", "jpeg", "png"])

# Handling the uploaded file
if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Store the image in a variable
    uploaded_image = image
    
    st.write("Image uploaded successfully!")

    final_img = run.main(uploaded_image)
    st.image(final_img, caption='Final Image', use_column_width=True)




else:
    st.write("Please upload an image file.")
