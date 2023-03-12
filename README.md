# RBE/CS549: P2 - Buildings built in minutes - SfM and NeRF

#


Implementing the original NERF method https://arxiv.org/abs/2003.08934

## Inputs:

Download the lego data for NeRF from the original author’s link https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a


Overall Inputs and Outputs of the Model are as shown below:




## Implementing Guidelines:

### For Training 
1. Go to the directory named Phase 2 of the file
2. Implement the below code for training the NeRF model on GPU

python3 Train_NeRF.py

After implementing above two steps for training, a checkpoint named model.ckpt will be saved, then you can implement the below code for the Testing.

### For Testing
1. Keep the same directory as you kept for Training
2. Implement the below code for testing the NeRF model on GPU

Code - python3 Test_NeRF.py

After implementing avove two steps for testing, an output video will be saved named as Output_NeRF.mp4 and a loss graph will be saved in NeRF_Output folder.