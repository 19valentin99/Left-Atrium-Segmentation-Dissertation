#**************************************************
#Author:  Valentin Craciun                        *
#Project: Heart Segmentation using Deep Learning  *
#Date:    08.04.2022                              *
#**************************************************

############################################################################################
#File Description:                                                                         #
#Convert images from .nii to .jpg (grey scale)                                             #
#(the main reason being that it is easier to train the model)                              #
############################################################################################


import os
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt

#setup file locations
IMAGES_FOLDER = "/Images"

IMAGES_NII_LOCATION  = "/Images_nii"
TRAINING_NII         = "/images_Training_nii"
TRAINING_LABBELS_NII = "/labels_Training_nii"
TESTING_NII          = "/images_Testing_nii"
TESTING_LABELS_NII   = "/labels_Testing_nii"

IMAGES_JPG_LOCATON   = "/Images_jpg"
TRAINING_JPG         = "/images_Training"
TRAINING_LABBELS_JPG = "/labels_Training"
TESTING_JPG          = "/images_Testing"
TESTING_LABELS_JPG   = "/labels_Testing"

CONVERT_IMAGES = True  # Chose to save or not the images

# initialise all the directories that will be used
def get_directories_paths():
    main_directory = os.getcwd()

    path_training_img_nii    = main_directory + IMAGES_FOLDER + IMAGES_NII_LOCATION + TRAINING_NII
    path_training_labels_nii = main_directory + IMAGES_FOLDER + IMAGES_NII_LOCATION + TRAINING_LABBELS_NII
    path_testing_img_nii     = main_directory + IMAGES_FOLDER + IMAGES_NII_LOCATION + TESTING_NII
    path_testing_labels_nii  =  main_directory+ IMAGES_FOLDER + IMAGES_NII_LOCATION + TESTING_LABELS_NII

    path_training_img_jpg    = main_directory + IMAGES_FOLDER + IMAGES_JPG_LOCATON + TRAINING_JPG
    path_training_labels_jpg = main_directory + IMAGES_FOLDER + IMAGES_JPG_LOCATON + TRAINING_LABBELS_JPG
    path_testing_img_jpg     = main_directory + IMAGES_FOLDER + IMAGES_JPG_LOCATON + TESTING_JPG
    path_testing_labels_jpg  = main_directory + IMAGES_FOLDER + IMAGES_JPG_LOCATON + TESTING_LABELS_JPG

    return path_training_img_nii, path_training_labels_nii, path_testing_img_nii, path_testing_labels_nii, \
           path_training_img_jpg, path_training_labels_jpg, path_testing_img_jpg, path_testing_labels_jpg


# convert the .nii fies to jpg (and save them)
def nii_to_jpg(from_location, to_location):
    images_name_list = sorted(Path(from_location).iterdir(), key=os.path.getmtime) #load the patients in order
    counter = 0                           # unique id name for each photo

    for idx, img in enumerate(images_name_list):          # .nii files are loaded in order
        print(f"Images Converted .nii to .jpg [{idx+1}/{len(images_name_list)}]", end="\r")
        img = str(img)

        img_3d_nii = sitk.ReadImage(img)   
        img_3d = sitk.GetArrayFromImage(img_3d_nii)

        for img in img_3d:
            plt.imsave(to_location + "/" + str(counter) + ".jpg", img, cmap="gray")

            counter += 1    # increment id
            

def main():
    # combine directories path for (./nii and .jpg) + for each type(train & test data + train & test labels)
    path_train_img_nii, path_train_lab_nii, path_test_img_nii, path_test_lab_nii,\
    path_train_img_jpg, path_train_lab_jpg, path_test_img_jpg, path_test_lab_jpg = get_directories_paths()
    print(".NII PATHS")
    print(path_train_img_nii)
    print(path_train_lab_nii)
    print(path_test_img_nii)
    print(path_test_lab_nii)
    print()
    print(".JPG PATHS")
    print(path_train_img_jpg)
    print(path_train_lab_jpg)
    print(path_test_img_jpg)
    print(path_test_lab_jpg)

    # take the images from each .nii directory and save them as .jpg in another directory 
    if CONVERT_IMAGES:
        print()
        print("Train Images")
        nii_to_jpg(path_train_img_nii, path_train_img_jpg)

        print()
        print("Train Labels")
        nii_to_jpg(path_train_lab_nii, path_train_lab_jpg)

        print()
        print("Test Images")
        nii_to_jpg(path_test_img_nii, path_test_img_jpg)

        print()
        print("Test Labels")
        nii_to_jpg(path_test_lab_nii, path_test_lab_jpg)
    

if __name__ == "__main__":
    main()