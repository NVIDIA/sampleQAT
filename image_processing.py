# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
from PIL import Image


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

_RESIZE_MIN = 256
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


def preprocess_imagenet(image, channels=3, height=224, width=224):
    """Pre-processing for Imagenet-based Image Classification Models:
        resnet50, vgg16, mobilenet, etc. (Doesn't seem to work for Inception)

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    assert img_data.shape[0] == channels

    for i in range(img_data.shape[0]):
        # Scale each pixel to [0, 1] and normalize per channel.
        img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    return img_data

def _smallest_size_at_least(height, width, resize_min):

    smaller_dim = np.minimum(float(height), float(width))
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = int(height * scale_ratio)
    new_width = int(width * scale_ratio)

    return new_height, new_width

def _central_crop(image, crop_height, crop_width):
    shape = image.shape
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    cropped_image = image[crop_top:crop_height+crop_top, crop_left:crop_width+crop_left]
    return cropped_image

def normalize_inputs(inputs):

    num_channels = inputs.shape[-1]

    if len(_CHANNEL_MEANS) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means_per_channel = np.reshape(_CHANNEL_MEANS, [1, 1, num_channels])
    # means_per_channel = tf.cast(means_per_channel, dtype=inputs.dtype)

    inputs = np.subtract(inputs, means_per_channel)/255.0

    return inputs


def preprocess_resnet50(image, channels=3, height=224, width=224):
    """Pre-processing for Imagenet-based Image Classification Models:
        resnet50 (resnet_v1_1.5 designed by Nvidia
    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the shape of the image.
    w, h= image.size

    new_height, new_width = _smallest_size_at_least(h, w, _RESIZE_MIN)

    # Image is still in WH format in PIL
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)
    # Changes to HWC due to numpy
    img_data = np.asarray(resized_image).astype(np.float32)
    # Do a central crop
    cropped_image = _central_crop(img_data, height, width)
    assert cropped_image.shape[0] == height
    assert cropped_image.shape[1] == width
    if len(cropped_image.shape) == 2:
        # For images without a channel dimension, we stack
        cropped_image = np.stack([cropped_image] * 3)
        return cropped_image
        # logger.debug("Received grayscale image. Reshaped to {:}".format(cropped_image.shape))

    normalized_inputs = normalize_inputs(cropped_image)
    cropped_image = np.transpose(normalized_inputs, [2, 0, 1])

    return cropped_image

def preprocess_inception(image, channels=3, height=224, width=224):
    """Pre-processing for InceptionV1. Inception expects different pre-processing
    than {resnet50, vgg16, mobilenet}. This may not be totally correct,
    but it worked for some simple test images.

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.BILINEAR)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    return img_data
