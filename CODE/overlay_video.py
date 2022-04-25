#**************************************************
#Author:  Valentin Craciun                        *
#Project: Heart Segmentation using Deep Learning  *
#Date:    08.04.2022                              *
#**************************************************

#################################################################################################################################
#File Description:                                                                                                              #
#Using the outputed overlay images (Original + Prediction + Ground Truth) generate a video  for each patient and store it.      #
#################################################################################################################################

import numpy as np
import cv2
from calculate_volume import images_per_patient
import os
from pathlib import Path


IMG_OVERLAY_DIR = "Images/OUTPUT_OVERLAY"                 # location of the overlay images
IMAGES_NII_PATH = "Images/Images_nii/images_Testing_nii"  # will use the .nii images in order to find out how many photos are per patient
                                                          # and display the volume per patient
VIDEO_DIRECTORY = "Images/Video_Overlay"

def generate_video(dir_path, imgs_per_patient):
    images_name_list = sorted(Path(dir_path).iterdir(), key=os.path.getmtime) # load images in the order they are on pc

    frames = []

    # keep track of image and pateint counter in order to display volume per patient
    img_counter = 0         
    patient_counter = 0

    for idx in(np.arange(0, len(images_name_list))):                           
        overlay_image = cv2.imread(str(images_name_list[idx]))    

        # resize the image to fit the video
        overlay_image = cv2.resize(overlay_image, (0,0), fx=0.5, fy=0.5)

        # add the image to the frames list
        frames.append(overlay_image)

        # if the number of images per patient is reached, then create a video
        if (img_counter == imgs_per_patient[patient_counter]-1):
            # create a video
            out = cv2.VideoWriter(str(VIDEO_DIRECTORY) + "/" + "Patient_" + str(patient_counter) + ".avi",\
                 cv2.VideoWriter_fourcc(*'DIVX'), 15, (overlay_image.shape[1], overlay_image.shape[0]))
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()
            frames = []
            img_counter = 0
            patient_counter += 1
        else:
            img_counter += 1



def main():
    imgs_per_patient = images_per_patient(IMAGES_NII_PATH)
    
    generate_video(dir_path=IMG_OVERLAY_DIR, imgs_per_patient=imgs_per_patient)


if __name__ == "__main__":
    main()