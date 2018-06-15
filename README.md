# ECE 285 


## Description
This project Adversarial Approaches for Generating Photorealistic Images of Landscapes developed by team Blended Pineapple Juice composed of Lucas Tindall, Christian Koguchi, Phat Phan

## Requirements
$ pip install --user torch torchvision tqdm matplotlib scikit-image pillow 
$ sh get_dataset.sh 

## Code organization
demo.ipynb -- Run a demo of our code ( reproduce Figure 7 of our report )   
train.ipynb -- Run the training of our model ( as described in Section 4)   
src_code/data_loader.py  -- Module for loading the monet2photo dataset    
src_code/models.py -- Module which defines our various network models (Cyclegan, Unet, PatchDiscriminator, Generator)    
src_code/cyclegan.py -- Script version of the train.ipynb notebook    
checkpoints/monet2photo/*.pth -- Saved model files for the CycleGAN generators and discriminators (G_AB, G_BA, D_A, D_B)   
get_dataset.sh -- Bash script to download the monet2photo dataset and place in the proper directory for use with our scripts 
