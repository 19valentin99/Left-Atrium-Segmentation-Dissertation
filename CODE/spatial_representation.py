#**************************************************
#Author:  Valentin Craciun                        *
#Project: Heart Segmentation using Deep Learning  *
#Date:    08.04.2022                              *
#**************************************************

#################################################################################
# File Description:                                                             #
# Display 3d representation of the left atrium segmentation                     #
#################################################################################

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk


IMG_OUT_PATH    = "Images/OUTPUT"                         # location of the images
IMAGES_NII_PATH = "Images/Images_nii/images_Testing_nii"  # will use the .nii images in order to find out how many photos are per patient
                                                          # and display the volume per patient

PREDICTION_POINT_CLOUD = False # choose if to visualize the point cloud of the prediction or the ground truth                                                         

# threshold values for contour mask
FOREGROUND = 1 
BACKGROUND = 0 

IMAGE_THRESHOLD = 240   # noise removal when reading from jpg

KERNEL =  np.ones((2,2),np.uint8) # kernel for the morphological operations (1 pixel wide contour) 


# return the number of images per patient
def images_per_patient(nii_dir_path):
    images_name_list = sorted(Path(nii_dir_path).iterdir(), key=os.path.getmtime) # load the patients in order
    heart_dimension_per_patient = []   # number of mri images per patient

    for img in images_name_list:       # .nii files are loaded in order
        img = str(img)

        img_3d_nii = sitk.ReadImage(img)  
        img_3d = sitk.GetArrayFromImage(img_3d_nii)

        heart_dimension_per_patient.append(img_3d.shape[0])
    
    return heart_dimension_per_patient

# return x, y, z coordinates of the segmentation per frame
def points_per_frame(img ,img_counter):
    x_countour, y_contour, z_contour = [], [], []
    threshold = np.where(img  > IMAGE_THRESHOLD, FOREGROUND, BACKGROUND)
    threshold = threshold.astype('uint8')
    contour = cv2.morphologyEx(threshold, cv2.MORPH_GRADIENT, KERNEL)
    
    for i,row in enumerate(contour):
        for j,element in enumerate(row):
            if element == 1:
                x_countour.append(j)
                y_contour.append(i)
                z_contour.append(img_counter)

    return x_countour, y_contour, z_contour


# load images & calculate volume
def representation(dir_path, imgs_per_patient):
    images_name_list = sorted(Path(dir_path).iterdir(), key=os.path.getmtime) # load images in the order they are on pc

    x, y, z = [], [], []   # x, y, z coordinates of the segmentation

    # keep track of image and patient counter in order to display volume per patient
    img_counter = 0       
    patient_counter = 0
    
    for idx in(np.arange(0,len(images_name_list), 3)): # select sets of 3 image
        print(f"Spatial Representation - Images Loaded [{img_counter}/ {np.sum(imgs_per_patient[:patient_counter+1])-1}]", end="\r")

        if PREDICTION_POINT_CLOUD:
            output = cv2.imread(str(images_name_list[idx+1]), cv2.IMREAD_GRAYSCALE)  # 2nd (prediction) 
            x_points, y_points, z_points = points_per_frame(output, img_counter)
        else:
            label  = cv2.imread(str(images_name_list[idx+2]), cv2.IMREAD_GRAYSCALE)  # 3rd (true output)    
            x_points, y_points, z_points = points_per_frame(label, img_counter)
        
        # add the points to the list
        x.extend(x_points)
        y.extend(y_points)
        z.extend(z_points)

        img_counter += 1 # increase the counter

        if img_counter >= np.sum(imgs_per_patient[:patient_counter+1]):
            patient_counter += 1
            
            # visualize the point cloud
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x, y, z , c=z, cmap='viridis')
            ax.set_title("3D representation of the left atrium segmentation - patient " + str(patient_counter))
            plt.show()

            #empty variables
            x, y, z = [], [], []

def main():
    imgs_per_patient = images_per_patient(IMAGES_NII_PATH)
    representation(dir_path=IMG_OUT_PATH, imgs_per_patient=imgs_per_patient)


if __name__ == "__main__":
    main()