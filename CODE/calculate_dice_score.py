#**************************************************
#Author:  Valentin Craciun                        *
#Project: Heart Segmentation using Deep Learning  *
#Date:    08.04.2022                              *
#**************************************************

#################################################################################################################################
#Reference for DICE SCORE:                                                                                                      #
#https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i  #
#Denis Berroeta                                                                                                                 #
#################################################################################################################################
#File Description:                                                                                                              #
#This file is to:                                                                                                               #
# 1) calculate dice score + to improve the accuracy of the predictions.                                                         #
# 2) output if wanted the overlay of the prediction and the ground truth over the original image.                               #
#################################################################################################################################

import numpy as np
import cv2
import os
from pathlib import Path


IMAGES_PATH    = "Images/OUTPUT"            # location of the images
IMAGES_OVERLAY = "Images/OUTPUT_OVERLAY"    # location of the overlay images

DATA_ENCHANCEMENT = True        # apply modification on the data
SAVE_CHANGES      = True        # save the modification
ADD_OVERLAY       = True        # save prediction/ ground_truth over the initial image

NORMAL_AVERAGE = 4.5            # below this threshold is when segmentation actually happens 
OUTPUT_MIN_CONFIDENCE_THR = 77  # threshold the outputs of the U-NET based on confidence 
LABEL_NOISE_REMOVAL_THR = 200   # some noise somehow was introduced in the labeled images (not all pixels are 255)

KERNEL = np.ones((5, 5), np.uint8) # for morphological operations

KERNEL_OVERLAY =  np.ones((4, 4),np.uint8)   # the contour of the given segmentation to have a width of 3 pixels
KERNEL_OVERLAY2 =  np.ones((3, 3),np.uint8)  # the contour of the predicted segmentation to have a width of 2 pixels


#load images & compute dice score
def compute_dice():
    images_name_list = sorted(Path(IMAGES_PATH).iterdir(), key=os.path.getmtime) # load images in the order they are on pc
    
    total_dice = 0      # sum of all the dice scores
    counter = 0         # total number of inputs

    for idx in(np.arange(0,len(images_name_list), 3)):                          # select sets of 3 image
        original = cv2.imread(str(images_name_list[idx]))                       # 1st image is the input
        output = cv2.imread(str(images_name_list[idx+1]), cv2.IMREAD_GRAYSCALE) # 2nd (prediction)
        label  = cv2.imread(str(images_name_list[idx+2]), cv2.IMREAD_GRAYSCALE) # 3rd (true output)

        # improve prediction and labelles quality
        if DATA_ENCHANCEMENT:
            # [STEP 1]
            # When the output should be blank (no segmentation is discovered)
            # the neural network outputs a whitish image of the input data.
            # After multiple experiments, I found out that the average in these
            # type of images is much higher than the average when secmendation is actually happening
            # therefore I will consider all the images that have a higher avg than the normal
            # segmentation to be 0 (no segmentation) 
            if (np.average(output) > NORMAL_AVERAGE):
                output = np.zeros((320,320,3))

            # [STEP 2]
            # helps with:
            # 1) output increase confidence (not all pixels are 255 or 0)
            # 2) removing noise after loading the images
            output = np.where(output > OUTPUT_MIN_CONFIDENCE_THR, 255, 0) # some pixels have small confidence
            label  = np.where(label  > LABEL_NOISE_REMOVAL_THR  , 255, 0) # some noise picked up from jpg reading

            # [STEP 3]
            # morphological opening/ closing in order to remove noise
            # that might have been introduced in the output and labelled images
            output= output.astype('uint8')
            output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, KERNEL)
            

            label = label.astype('uint8')
            label = cv2.morphologyEx(label, cv2.MORPH_CLOSE, KERNEL)
        else:
            output = np.where(output > OUTPUT_MIN_CONFIDENCE_THR, 255, 0) # some pixels have small confidence
            label  = np.where(label  > LABEL_NOISE_REMOVAL_THR  , 255, 0) # some noise picked up from jpg reading

        # save the changes
        if SAVE_CHANGES:                                           
            cv2.imwrite(str(images_name_list[idx])  , original.astype(np.uint16))
            cv2.imwrite(str(images_name_list[idx+1]), output.astype(np.uint16))
            cv2.imwrite(str(images_name_list[idx+2]), label.astype(np.uint16))


        # overlay the ground truth and prediction over the original image
        if ADD_OVERLAY:
            save_overlay(original, output, label, counter)
            

        total_dice += dice(output, label) # compute sum of dice scores
        counter += 1                                

    return total_dice / counter           # calculate avg dice score


# compute the contour of the ground truth and the prediction
# and overlay them over the original image
def save_overlay(original, output, label, counter):
    overlay = np.copy(original)

    overlay_label = np.where(label  > LABEL_NOISE_REMOVAL_THR, 1, 0)
    overlay_label = overlay_label.astype('uint8')
    overlay_label = cv2.morphologyEx(overlay_label, cv2.MORPH_GRADIENT, KERNEL_OVERLAY)

    output_copy = np.copy(output)  # before applying image enchantment, the output is 3d (even if it has only one channel)
    if output_copy.ndim == 3:
        output_copy = output_copy[:,:,2]

    overlay_prediction = np.where(output_copy  > OUTPUT_MIN_CONFIDENCE_THR, 1, 0)
    overlay_prediction = overlay_prediction.astype('uint8')
    overlay_prediction = cv2.morphologyEx(overlay_prediction, cv2.MORPH_GRADIENT, KERNEL_OVERLAY2)
    
    overlay[overlay_label == 1]      = [0, 0, 255] # red (BGR colour format)
    overlay[overlay_prediction == 1] = [0, 255, 0] # green (BGR colour format)

    cv2.imwrite(IMAGES_OVERLAY + "/" +str(counter) + ".jpg", overlay.astype(np.uint16))


# calculate dice score
def dice(pred, true, k = 255):
    intersection = np.sum(pred[true==k]) * 2.0

    if (np.sum(pred) + np.sum(true)) != 0:   # when there is some segmentation discovered
        dice = intersection / (np.sum(pred) + np.sum(true))
        return dice
    else:                                    # when there is no pattern to be discovered
                                             # and segmentation was right, return 1
        return 1


# run the algorithm
def main():
    avg_dice = compute_dice()
    print("Average Dice Score: ", avg_dice)
    print("==========")


if __name__ == "__main__":
    main()
