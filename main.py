# pip install timm
from PIL import Image
import numpy as np
import torch
from transformers import DetrForSegmentation, DetrFeatureExtractor
import io
import os
from transformers.models.detr.feature_extraction_detr import rgb_to_id

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# preload model before calling function
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
model = model.to(device)

# input on streamlit
input_img = 'zane_test.jpg'
input_bg = 'rome.jpg'

input_img_dir = 'input_images/'
input_bg_dir = 'background_images/'

image = Image.open(os.path.join(input_img_dir, input_img))
bg_image = Image.open(os.path.join(input_bg_dir, input_bg))

def change_bg(image, bg_image, model, feature_extractor):
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

    # way of identifying bg: category id == 199
    id_bg = -1
    for x in result['segments_info']:
        if x['category_id'] == 199:
            id_bg = x['id']

    # seg_fg = (panoptic_seg_id != id_bg)
    seg_bg = (panoptic_seg_id == id_bg)

    bg_image = bg_image.resize((seg_bg.shape[1], seg_bg.shape[0]))
    image = image.resize((seg_bg.shape[1], seg_bg.shape[0]))

    image, bg_image = np.array(image), np.array(bg_image)
    image[seg_bg] = bg_image[seg_bg]

    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Image.fromarray(image).save(os.path.join(output_dir , 'output.jpg'))

    return image

# def deepdream(image):


if __name__ == "__main__":
    bg_replaced_image = change_bg(image, bg_image, model, feature_extractor)
