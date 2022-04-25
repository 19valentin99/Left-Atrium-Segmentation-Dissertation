#**************************************************
#Author:  Valentin Craciun                        *
#Project: Heart Segmentation using Deep Learning  *
#Date:    08.04.2022                              *
#**************************************************

#################################################################################################################################
#File Description:                                                                                                              #
#After the segmentation is done, this program is to calculate the volume of the segmentation per patient in cm^3 and ml.        #
#################################################################################################################################

import numpy as np
import cv2
import os
from pathlib import Path
import SimpleITK as sitk


IMG_OUT_PATH    = "Images/OUTPUT"                         # location of the images
IMAGES_NII_PATH = "Images/Images_nii/images_Testing_nii"  # will use the .nii images in order to find out how many photos are per patient
                                                          # and display the volume per patient


SEGMENTATION_VALUE = 255 # segmentation value on pixel (255 - segmentation)
BACKGROUND_VALUE   = 0   # the parts that are not included in the segmentation  (0 - background)  

N_CHANNELS = 3           # the number of channels is 3 because the images 
IMAGE_THRESHOLD = 240
MM_TO_ML = 0.001         # convert from mm^3 to ml

#MRI Voxel Size in mm (the size of voxel in mri)
VOXEL_SPACING_ON_X = 1 
VOXEL_SPACING_ON_Y = 1
VOXEL_SPACING_ON_Z = 1

# return the number of images per patient
def images_per_patient(nii_dir_path):
    images_name_list = sorted(Path(nii_dir_path).iterdir(), key=os.path.getmtime) # load the patients in order
    
    heart_dimension_per_patitent = []

    for img in images_name_list:          # .nii files are loaded in order
        img = str(img)

        img_3d_nii = sitk.ReadImage(img)  
        img_3d = sitk.GetArrayFromImage(img_3d_nii)

        heart_dimension_per_patitent.append(img_3d.shape[0])
    
    return heart_dimension_per_patitent


# load images & calculate volume
def calculate_volume(dir_path, imgs_per_patient):
    images_name_list = sorted(Path(dir_path).iterdir(), key=os.path.getmtime) # load images in the order they are on pc

    volume_output = np.zeros(len(imgs_per_patient))
    true_volume = np.zeros(len(imgs_per_patient))

    # keep track of image and pateint counter in order to display volume per patient
    img_counter = 0         
    patient_counter = 0
    
    for idx in(np.arange(0,len(images_name_list), 3)):                           # select sets of 3 image
        output = cv2.imread(str(images_name_list[idx+1]), cv2.IMREAD_GRAYSCALE)  # 2nd (prediction) 
        label  = cv2.imread(str(images_name_list[idx+2]), cv2.IMREAD_GRAYSCALE)  # 3rd (true output)  

        # make sure that the labels are 255 or 0  (due to image compression))
        output = np.where(output > IMAGE_THRESHOLD, SEGMENTATION_VALUE, BACKGROUND_VALUE)
        label   = np.where(label > IMAGE_THRESHOLD, SEGMENTATION_VALUE, BACKGROUND_VALUE)
        
        # calculate the volume per picture
        volume_output[patient_counter] += (np.sum(output)/ SEGMENTATION_VALUE) 
        true_volume[patient_counter]   += (np.sum(label) / SEGMENTATION_VALUE) 

        img_counter += 1

        if img_counter >= np.sum(imgs_per_patient[:patient_counter+1]):
            patient_counter += 1

    return volume_output, true_volume


def main():
    imgs_per_patient = images_per_patient(IMAGES_NII_PATH)
    out_vol, true_vol = calculate_volume(dir_path=IMG_OUT_PATH, imgs_per_patient=imgs_per_patient)

    # display the volume per patient
    for idx in range(out_vol.shape[0]):
        print(f"Patient({idx})",
              f"Output volume: {out_vol[idx]} mm^3 || ",
              f"True volume: {true_vol[idx]} mm^3")

    print("==========")

    for idx in range(out_vol.shape[0]):
        print(f"Patient({idx}) ",
              f"Output volume: {round(out_vol[idx] * MM_TO_ML, 3)} ml || ",
              f"True volume: {round(true_vol[idx] * MM_TO_ML, 3)} ml")


if __name__ == "__main__":
    main()