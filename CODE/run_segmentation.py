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

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET
import time
import matplotlib.pyplot as plt
import pathlib

import calculate_dice_score
import calculate_volume
import overlay_video
import spatial_representation

from utils import get_loaders

# run spatial representation for each test patient (point cloud of the segmentation)
SPATIAL_REPRESENTATION = True

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 3      
NUM_WORKERS = 2
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
PIN_MEMORY = True

LOAD_MODEL = True  # if the model has been already trained (can skip the training)

# image directories 
CURRENT_DIRECTORY = str(pathlib.Path().resolve())
TRAIN_IMG_DIR  = CURRENT_DIRECTORY  + "/Images/Images_jpg/images_Training"
TRAIN_MASK_DIR = CURRENT_DIRECTORY  + "/Images/Images_jpg/labels_Training"
TEST_IMG_DIR   = CURRENT_DIRECTORY  + "/Images/Images_jpg/images_Testing"
TEST_MASK_DIR  = CURRENT_DIRECTORY  + "/Images/Images_jpg/labels_Testing"

#location to save test' prediciton
OUTPUT = CURRENT_DIRECTORY  + "/Images/OUTPUT"


def train_model(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def test_model(loader, model, loss_fn):
    loop = tqdm(loader)
    
    counter = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, targets) in enumerate(loop):

            data = data.to(device=DEVICE)
            target = targets.float().unsqueeze(1).to(device=DEVICE)

            data_to_save = torch.squeeze(data)
            data_to_save = data_to_save.cpu().data.numpy()
            plt.imsave(OUTPUT + "/" + str(counter) + ".jpg", data_to_save, cmap="gray")
            counter += 1
            
            prediction = model(data)
            prediction_to_save = torch.squeeze(prediction)
            prediction_to_save = prediction_to_save.cpu().data.numpy()
            plt.imsave(OUTPUT + "/" + str(counter) + ".jpg", prediction_to_save, cmap="gray")
            counter += 1


            target_to_save = torch.squeeze(target)
            target_to_save = target_to_save.cpu().data.numpy()
            plt.imsave(OUTPUT + "/" + str(counter) + ".jpg", target_to_save, cmap="gray")
            counter += 1


# return the transforms that need to be done for the training and testing sets
def train_test_transforms():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    return train_transform, test_transform

# run the training/ testing algorithms
def main():
    start_time = time.time()                                    # start the timer
    
    train_transform, test_transform = train_test_transforms()   # get train/ test data transforms

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)      # create model
    loss_fn = nn.BCEWithLogitsLoss()                            # initialise loss function
    optimizer= optim.Adam(model.parameters(), lr=LEARNING_RATE) # initialize optimizer

    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )


    scaler = torch.cuda.amp.GradScaler()
    
    if not LOAD_MODEL:                                                 # if load model is false (train)
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch: [{epoch+1}/{NUM_EPOCHS}]")
            train_model(train_loader, model, optimizer, loss_fn, scaler)

        trained_model ={                                               # save model at the end of training
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(trained_model, "trained.pt")                       
    else:                                                              # else (load the model)
        try:
            trained = torch.load("trained.pt")
            model.load_state_dict(trained['model_state_dict'])
            optimizer.load_state_dict(trained['optimizer_state_dict'])
        except:
            print("no model found")
            return


    # testing the model (perform segmentation on unseen data)
    print("Testing the model")
    test_model(test_loader, model, loss_fn)

    calculate_dice_score.main() # compute dice score (+ improvements) and save prediction overlay
    calculate_volume.main()     # compute volume per patient

    if calculate_dice_score.ADD_OVERLAY:
        overlay_video.main()        # create overlay per patient 

    print("--- %s seconds ---" % (time.time() - start_time))    # stop the timer


    if SPATIAL_REPRESENTATION:  
        spatial_representation.main() # create spatial representation of segmentation


if __name__ == "__main__":
    main()