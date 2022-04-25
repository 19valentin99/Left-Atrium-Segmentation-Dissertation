#**************************************************
#Author:  Valentin Craciun                        *
#Project: Heart Segmentation using Deep Learning  *
#Date:    08.04.2022                              *
#**************************************************

##################################################################################
# File Description:                                                              #
# This file is used to create the placeholder folders for:                       #                 
# 1) .nii files (training and testing data)                                      #
# 2) .jpg files (training and testing data)                                      #
# 3) Output folder                                                               #                                    
# 3) Overlay Images                                                              #    
# 4) Videos using the Overlay Images                                             #
##################################################################################
import os
import pathlib

CURRENT_DIRECTORY = str(pathlib.Path().resolve())

IMAGES = CURRENT_DIRECTORY+ "/Images"

IMAGES_JPG = IMAGES + "/Images_jpg"
IMAGES_JPG_images_training = IMAGES_JPG + "/images_Training"
IMAGES_JPG_labels_training = IMAGES_JPG + "/labels_Training"
IMAGES_JPG_images_testing  = IMAGES_JPG + "/images_Testing"
IMAGES_JPG_labels_testing  = IMAGES_JPG + "/labels_Testing"

IMAGES_NII = IMAGES + "/Images_nii"
IMAGES_NII_images_training = IMAGES_NII + "/images_Training_nii"
IMAGES_NII_labels_training = IMAGES_NII + "/labels_Training_nii"
IMAGES_NII_images_testing  = IMAGES_NII + "/images_Testing_nii"
IMAGES_NII_labels_testing  = IMAGES_NII + "/labels_Testing_nii"

OUTPUT         = IMAGES + "/OUTPUT"
OUTPUT_OVERLAY = IMAGES + "/OUTPUT_OVERLAY"
OUTPUT_VIDEO   = IMAGES + "/Video_Overlay"


CREATE =  [IMAGES, IMAGES_JPG,]
CREATE += [IMAGES_JPG_images_training, IMAGES_JPG_labels_training,]
CREATE += [IMAGES_JPG_images_testing, IMAGES_JPG_labels_testing,]
CREATE += [IMAGES_NII, IMAGES_NII_images_training, IMAGES_NII_labels_training,]
CREATE += [IMAGES_NII_images_testing, IMAGES_NII_labels_testing,]
CREATE += [OUTPUT, OUTPUT_OVERLAY, OUTPUT_VIDEO]

# run the file generation
def main():
    for folder in CREATE:
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == "__main__":
    main()