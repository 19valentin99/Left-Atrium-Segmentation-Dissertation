#**************************************************
#Author:  Valentin Craciun                        *
#Project: Heart Segmentation using Deep Learning  *
#Date:    08.04.2022                              *
#**************************************************

########################################################################################################################################################
# Inspired from:                                                                                                                                       #
# Original paper: https://arxiv.org/abs/1505.04597                                                                                                     #
# GitHub repository: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet#
########################################################################################################################################################

import numpy as np
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class SegmendationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = sorted(Path(image_dir).iterdir(), key=os.path.getmtime)    
        self.mask_dir = sorted(Path(mask_dir).iterdir(), key=os.path.getmtime)      

        self.transform = transform

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(self.image_dir[index]).convert("L"), dtype=np.float32)  #L because the image will be gray scale
        mask = np.array(Image.open(self.mask_dir[index]).convert("L"), dtype=np.float32)  #L because mask will be gray scale

        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            image = augmentations["image"]
            mask =  augmentations["mask"]

        return image, mask