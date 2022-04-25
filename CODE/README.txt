****************************************
*INITIALISE THE PROGRAMMING ENVIORNMENT*
****************************************
(This will create a python environment with all the libraries that I used for the project)

OPTION 1:
1)	conda env create -f environment.yml
2)	conda activate torch

OPTION 2:
Install the libraries that I used manually in your own environment:
1)	PyTorch (view https://pytorch.org/  for more information – based on the GPU version a specific version of cuda might be required)
2)	NumPy 
3)	OpenCv 2 (for image processing)
4)	Pathlib (for initializing folders)
5)	SimpleITK (for reading the .nii imaged)
6)	Matplotlib (for saving images and visualize 3d plots of the left atrium)
7)	Pillow (reading .jpg images)
8)	Albumentations (augment the images – mean-0 and std-1)
9)	Tqdm (add loading bars in the terminal)



*****************
*RUN THE PROGRAM*
*****************
Note: if another dataset (other than the Left Atrium dataset) is used, some parameters from inside of the files will need to be changed (for example, from ``run_segmentation.py”: the image width/ height or loading the model from memory/ generating a new one)
0) 	You can use the following link to download a pre-trained model (https://www.dropbox.com/s/akohgos8j9lmfo5/trained.pt?dl=0). It needs to be placed in the main folder with the other python files.
	Furthermore, in order to use a pre-trained model, you have to set the variable ``LOAD MODEL" in run_segmentation.py to ``True".
1)	python generate_folder_structure.py
	a.	will generate all the required folder for the user to add the training dataset (.nii images and segmentations)
	b.    use the following link (https://www.dropbox.com/s/jihxggrx754o18c/Images_nii.zip?dl=0) in order to download the dataset from dropbox (you can replace the generated directory with the downloaded one).
2)	python convert_nii_to_jpg.py
	a.	will convert the training .nii images/ segmentations in .jpg images, saving them in the appropriate folders
3)	python run_segmentation.py
	a.	before running the segmentation make sure that all the variables in the files are set accordingly (if a model was not downloaded before or the model was no trained before, turn off the LOAD_MODEL variable from run_segmentation.py)
	b.	using this command is going to train the model and generate a loading file (which can be used in the feature to replace the training part)
	c.	test the model against the dataset, generating a 
		i.	dice score
		ii.	predicted volume/ actual volume
		iii.	output images (heart, prediction, true output)
		iv.	overlay images (over the heart input, overlay the contours of the prediction in green and true output in red) [if the variable ``ADD_OVERLAY” form ``calculate_dice_score.py” is set to ``True”]
		v.	generate videos for each patient with the overlay images along the 3d representation of the heart [if the variable ``ADD_OVERLAY” form ``calculate_dice_score.py” is set to ``True”]
		vi.	generate interactive 3D point cloud representation of the left atrium [if ``SPATIAL_REPRESENTATION” variable from ``run_segmentation.py” is set to True]
