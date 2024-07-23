# pip install timm
from PIL import Image
import cv2 as cv
import numpy as np
import torch
from transformers import DetrForSegmentation, DetrFeatureExtractor
import io
import os
from transformers.models.detr.feature_extraction_detr import rgb_to_id
from pytorch_deepdream import deepdream
# import streamlit as st

OUTPUT_DIR = 'output/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# preload model before calling function
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
model = model.to(device)

# # input on streamlit, these 6 lines of code are not necessary for streamlit app
# # only change if you want to run it locally (python3 run.py)
# input_img = 'zane_test.jpg'
# input_bg = 'rome.jpg'
# INPUT_IMG_DIR = 'input_images/'
# INPUT_BG_DIR = 'background_images/'
# image = Image.open(os.path.join(INPUT_IMG_DIR, input_img))
# bg_image = Image.open(os.path.join(INPUT_BG_DIR, input_bg))

def resize(img):
    DESIRED_RATIO = 640

    width, height = img.size
    bigger = width if width > height else height
    divide_ratio = bigger / DESIRED_RATIO

    img = img.resize((int(width/divide_ratio), int(height/divide_ratio)))
    return img

def change_bg(image, bg_image):
    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device)

    # forward pass
    outputs = model(**inputs)

    # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
    processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

    # the segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)

    # retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb_to_id(panoptic_seg)

    print(result)

    # identify foreground as person object only
    id_fg = []
    categories_to_include = [1, 16, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 75, 77, 84, 87, 88, 89, 90, 91] # edit based on what you want to stay in the foreground
    for x in result['segments_info']:
        if x['category_id'] in categories_to_include:
            id_fg.append(x['id'])

    seg_fg = np.isin(panoptic_seg_id, id_fg)
    seg_bg = np.logical_not(np.isin(panoptic_seg_id, id_fg))

    bg_image = bg_image.resize((seg_bg.shape[1], seg_bg.shape[0]))
    image = image.resize((seg_bg.shape[1], seg_bg.shape[0]))

    image, bg_image = np.array(image), np.array(bg_image)
 
    # # delete this after testing vvv
    # image[seg_fg] = 0
    # return image

    image[seg_bg] = bg_image[seg_bg]

    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
    # Image.fromarray(image).save(os.path.join(OUTPUT_DIR , 'bg_replaced_output.jpg'))

    return image

def resize_bg_image(image, bg_image):    # Get the dimensions of the input image
    # Check if the image is portrait or landscape
    if image.height > image.width:
        # Image is portrait, resize the background image vertically
        bg_aspect_ratio = bg_image.width / bg_image.height

        new_height = image.height

        # Resize the background image
        resized_bg_image = bg_image.resize((int(bg_aspect_ratio * new_height), new_height))
        
        # Calculate coordinates to crop the background image to the size of the input image
        left = (resized_bg_image.width - image.width) // 2
        top = 0
        right = left + image.width
        bottom = image.height
        
        # Crop the background image
        cropped_bg_image = resized_bg_image.crop((left, top, right, bottom))
    else:
        # Image is landscape, don't resize the background image
        cropped_bg_image = bg_image

    return cropped_bg_image
    

def call_change_bg(image, background_image):

    background_image = resize_bg_image(image, background_image)
    bg_replaced_image_nparray = change_bg(image, background_image)
    bg_replaced_image = Image.fromarray(bg_replaced_image_nparray)
    return bg_replaced_image

def call_deepdream(image):
    IMG_WIDTH_DEEPDREAM = 1000
    NUM_ITERS_DEEPDREAM = 2
    LR_DEEPDREAM = 0.09

    image.save(os.path.join(OUTPUT_DIR, 'bg_replaced_output.jpg'))
    output_nparray = deepdream.change_to_deepdream(os.path.join(OUTPUT_DIR , 'bg_replaced_output.jpg'), IMG_WIDTH_DEEPDREAM, NUM_ITERS_DEEPDREAM, LR_DEEPDREAM)

    if output_nparray.dtype != np.uint8:
        output_nparray = (output_nparray*255).astype(np.uint8)
    # cv.imwrite(os.path.join(OUTPUT_DIR, 'final_output.jpg'), output_nparray[:, :, ::-1])

    # return Image.open(os.path.join(OUTPUT_DIR, 'final_output.jpg'))

    # to return an image:
    output_nparray_uint8 = output_nparray.astype(np.uint8)
    res = Image.fromarray(output_nparray_uint8)
    res.save(os.path.join(OUTPUT_DIR, 'final_output.jpg'))
    return res

# if __name__ == "__main__":
#     main(image, bg_image)