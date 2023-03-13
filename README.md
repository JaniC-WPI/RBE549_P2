# RBE/CS549: P2 - Buildings built in minutes - SfM and NeRF

Implementing the SfM Pipeline

# cd <folder where codes are added>
-- open Terminal
-- run the following
> python3 Wrapper.py - <path\to\data\folder> - <path\to\save\output>

-- Outputs are saved in the folder Phase_1\Data\IntermediateOutputImages\

Implementing the original NERF method https://arxiv.org/abs/2003.08934

## Inputs:

Download the lego data for NeRF from the original author’s link https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a


Below is an Overall Overview of our Neural Radiance Field scene representation and differentible rendering procedure:-





## Implementing Guidelines:

### For Training 
1. Go to the directory named Phase 2 of the file
2. Implement the below code for training the NeRF model on GPU
    
    python3 Train_NeRF.py

After implementing above two steps for training, a checkpoint named model.ckpt will be saved, then you can implement the below code for the Testing.

### For Testing
1. Keep the same directory as you kept for Training
2. Implement the below code for testing the NeRF model on GPU

    python3 Test_NeRF.py

After implementing avove two steps for testing, an output video will be saved named as Output_NeRF.mp4 and a loss graph will be saved in NeRF_Output folder.



## References

1. https://arxiv.org/pdf/2003.08934.pdf
2. https://github.com/facebookresearch/neuralvolumes
3. https://rbe549.github.io/spring2023/proj/p2/
