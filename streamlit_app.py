import streamlit as st
from PIL import Image
import run
import os
import io

# CSS 
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #FFD1DC;
#     }
#     .horizontal-radio {
#         display: flex;
#         gap: 10px;
#     }
#     .horizontal-radio div {
#         margin-right: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

BG_DIR = "background_images/"

# add more if u want
background_options = {
    "Greece": "Greece.jpg",
    "Switzerland": "Switzerland.JPG",
    "Denmark": "Denmark.jpg",
    "Rome": "rome.jpg"
}

# title and description
st.title("Image Background Change and Style Transform")
st.write("Upload an image to change its background and transform its style.")
background_choice = st.radio("Choose a background:", list(background_options.keys()), horizontal=True)

uploaded_file = st.file_uploader("Choose an input image for background change and style transform.", type=["jpg", "jpeg", "png"])


# handling the uploaded file
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    # display words / image
    st.image(input_image, caption='Uploaded Image', use_column_width=True)
    st.write("Image uploaded successfully!")

    # load the selected background image
    background_image_path = os.path.join(BG_DIR, background_options[background_choice])
    background_image = Image.open(background_image_path)
    
    # run code
    final_img = run.main(input_image, background_image)
    
    # display final image
    st.image(final_img, caption='Final Image', use_column_width=True)

    # Convert final image to bytes
    img_byte_arr = io.BytesIO()
    final_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Add a download button for the final image
    st.download_button(
        label="Download Final Image",
        data=img_byte_arr,
        file_name="final_image.png",
        mime="image/png"
    )
else:
    st.write("Please upload an image file.")
