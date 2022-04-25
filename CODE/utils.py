########################################################################################################################################################
# Credits to:                                                                                                                                          #
# original paper: https://arxiv.org/abs/1505.04597                                                                                                     #
# github repository: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet#
########################################################################################################################################################

import torch
from dataset import SegmendationDataset
from torch.utils.data import DataLoader

def get_loaders(
    train_dir,
    train_maskdir,
    test_dir,
    test_maskdir,
    batch_size,
    train_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = SegmendationDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_ds = SegmendationDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader
