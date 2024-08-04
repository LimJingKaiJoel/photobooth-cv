import streamlit as st
from PIL import Image
import run
import os
import io
import numpy as np

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

# with open("styles.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

BG_DIR = "background_images/"

# add more if u want
background_options = {
    "Greece": "Greece.jpg",
    "Switzerland": "Switzerland.JPG",
    "Denmark": "Denmark.jpg",
    "Rome": "rome.jpg",
    "Paris": "Paris.jpg",
    "Hogwarts": "Hogwarts.jpg",
    "Rune": "Rune.jpg",
    "Cyberpunk": "Cyberpunk.jpg",
    "SMU": "SMU.jpg",
    "Cherry Blossoms": "Cherry-Blossoms.jpg",
    "SMUBIA1": "SMUBIA1.jpg",
    "SMUBIA2": "SMUBIA2.jpg",
    "SMUBIA3": "SMUBIA3.jpg",
    "SMUBIA4": "SMUBIA4.jpg",
}

# not in use until we have more than one style
style_options = {
    "Deepdream" : "deepdream"
}

# title and description
st.title("Transform your photo!")
st.write("Upload an image to change its background and transform its style.")
background_choice = st.radio("Choose a background:", list(background_options.keys()), horizontal=True)

# load the selected background image
background_image_path = os.path.join(BG_DIR, background_options[background_choice])
background_image = Image.open(background_image_path)

st.image(background_image, caption='Selected Background Image', use_column_width=True)

style_choice = st.radio("Choose a style:", list(style_options.keys()), horizontal=True)

uploaded_file = st.file_uploader("Choose an input image for background change and style transform.", type=["jpg", "jpeg", "png"])

# handling the uploaded file
if uploaded_file is not None:
    input_image = run.resize(Image.open(uploaded_file))
    
    # display words / image
    st.image(input_image, caption='Uploaded Image', use_column_width=True)
    st.write("Image uploaded successfully!")
    
    # run code
    bg_replaced_img = run.call_change_bg(input_image, background_image)

    # display bg image
    st.image(bg_replaced_img, caption=f'Image set in {background_choice}')
    img_byte_arr_bg = io.BytesIO()
    bg_replaced_img.save(img_byte_arr_bg, format='PNG')
    img_byte_arr_bg = img_byte_arr_bg.getvalue()

    # # test bg
    # bg_segment = run.change_bg(input_image, background_image)
    # output_nparray_uint8 = bg_segment.astype(np.uint8)
    # final_img = Image.fromarray(output_nparray_uint8)
    
    # Convert images to bytes
    final_img = run.call_deepdream(bg_replaced_img)
    img_byte_arr_final = io.BytesIO()
    final_img.save(img_byte_arr_final, format='PNG')
    img_byte_arr_final = img_byte_arr_final.getvalue()

    # Add a download button for the final image
    st.download_button(
        label="Download Background Replaced Image",
        data=img_byte_arr_bg,
        file_name="bg_replaced_image.png",
        mime="image/png"
    )

    # display final image
    st.image(final_img, caption='Final Image', use_column_width=True)

    # Add a download button for the final image
    st.download_button(
        label="Download Final Image",
        data=img_byte_arr_final,
        file_name="final_image.png",
        mime="image/png"
    )
else:
    st.write("Please upload an image file.")
