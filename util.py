from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    append_text_underneath_image
)
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

def save_map(observations, info, images):
    im = observations["rgb"]
    top_down_map = draw_top_down_map(info, im.shape[0])
    output_im = np.concatenate((im, top_down_map), axis=1)
    output_im = append_text_underneath_image(
        output_im, observations["instruction"]["text"]
    )
    images.append(output_im)

def rgb_to_base64(rgb_image):
    """
    Convert a RGB image to base64 string.
    """
    buffered = BytesIO()
    im = Image.fromarray(rgb_image)
    im.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
