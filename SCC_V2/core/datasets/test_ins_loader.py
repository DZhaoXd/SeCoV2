# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 16:48:36 2025

@author: 15642
"""

import torch
import numpy as np
import os
import json
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from cityscapes import CityscapesInsDataSet  # Assuming the dataset class is in this module
from transform import Resize, ToTensor, Normalize, Compose

import torch
import numpy as np
import cv2
import os
import random
from matplotlib import colors

def random_color_generator(num_colors):
    """
    Generate a list of random colors using HSV space, then convert them to RGB.
    :param num_colors: Number of unique colors to generate.
    :return: List of RGB colors.
    """
    # Create a color palette with unique hues for each color
    colors_list = []
    for i in range(num_colors):
        # Generate a hue value between 0 and 1 (hue range in HSV)
        hue = i / float(num_colors)
        
        # Convert from HSV to RGB
        rgb = colors.hsv_to_rgb([hue, 1, 1])  # Saturation = 1, Value = 1
        # Convert from [0, 1] to [0, 255] range and round
        colors_list.append([int(x * 255) for x in rgb])
        
    return colors_list 

def test_cityscapes_dataset(data_root, data_list, output_dir, transform=None, debug=False):
    # Initialize dataset
    dataset = CityscapesInsDataSet(
        data_root=data_root,
        data_list=data_list,
        transform=transform,
        debug=debug
    )
    
    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate a random color palette for each channel
    color_map = random_color_generator(50)
    
    # Iterate through the dataset
    for i, (image, sam_masks, sam_psd_labels, sam_gt_labels, name) in enumerate(dataloader):
        # sam_masks shape is [b, X, w, h] -> batch_size, num_channels, width, height
        batch_size, num_channels, height, width = sam_masks.shape

        # Iterate over each sample in the batch
        for batch_idx in range(batch_size):
            # Create an empty image to store the colored masks
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)  # Shape: (H, W, 3)

            # Iterate through each mask channel (X)
            for x in range(num_channels):
                mask = sam_masks[batch_idx, x] > 0  # Create binary mask for the current channel
                
                # Assign color to the mask area (using the color_map for each X)
                color = color_map[x % len(color_map)]  # Ensure we don't go out of bounds
                color_mask[mask.cpu().numpy()] = color  # Apply the color to the non-zero area of the mask

            # Convert image to numpy for concatenation and resizing
            image_numpy = np.array(image[batch_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # Convert from [0, 1] to [0, 255]
            
            # Ensure image and color_mask have the same size (resize if necessary)
            if image_numpy.shape[:2] != color_mask.shape[:2]:
                color_mask = cv2.resize(color_mask, (image_numpy.shape[1], image_numpy.shape[0]))

            # Concatenate the image and color_mask horizontally
            combined_image = np.hstack((image_numpy, color_mask))

            # Save the combined image
            name_ = name[batch_idx].split('/')[-1]
            output_path = os.path.join(output_dir, f"{name_}_combined.jpg")
            cv2.imwrite(output_path, combined_image)  # Save as a single image

            # Optional: Print the name of the image and mask dimensions
            print(f"Processed {name[batch_idx]} - Combined Image Shape: {combined_image.shape}")

        
    print("Testing completed.")

# Example usage
data_root = '/data/zd/SeCoV2/data/UDA/cityscapes/train'
data_list = '/data/zd/SeCoV2/data/UDA/G2C/DTST_PL/CS_no_thr/train_sam_mask_infos.json'
output_dir = './debug_directory'

trans = Compose([
    Resize((512, 1024), resize_label=False),
    ToTensor(),
    Normalize(mean=(0,0,0), std=(1,1,1), to_bgr255=False)
])

test_cityscapes_dataset(data_root, data_list, output_dir, transform=trans)
