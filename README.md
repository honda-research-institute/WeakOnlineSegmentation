# Weakly Supervised Online Segmentation


This is the code for the paper "Weakly Supervised Online Segmentation in Multiview Instructional Videos" accepted in CVPR 22.


Main Software Requirements:


+ PyTorch 1.9

+ Python 3.8

+ numpy among others

+ Ubuntu 18





********************************************************************************************************************
Instructions to reproduce the online segmentation results on the Beakfast dataset using iDT features:
********************************************************************************************************************
 

# Data Preparation

0-1- Download the pre-computed iDT from the third party link used in [1]: https://uni-bonn.sciebo.de/s/wOxTiWe5kfeY4Vd

0-2- Extract the content of the "data/features/"  inside our defined "/data/features/" directory. (should be in .npy format)

0-3- Extract the content of the "data/groundTruth/"  inside our defined "/data/groundTruth/" directory. (already done)

0-4- Extract the content of the "data/transcripts/"  inside our defined "/data/transcripts/" directory. (already done)

# Execution

1-1- Go to utils/options.py and change the parameters if desired. (e.g.multiview inference technique)

1-2- Type the following command line in terminal: python train.py

# Citation
 
 Please cite our paper if you use our code:
 
"Reza Ghoddoosian, Isht Dwivedi, Nakul Agarwal, Chiho Choi, and Behzad Dariush. Weakly-supervised online action segmentation in multi-view instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13780–13790, 2022"

# References

We used the source codes of the papers below in our implementation. Please consider citing them if you use our code.

[1] A. Richard, H. Kuehne, A. Iqbal, J. Gall: NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning in IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2018

[2]-Jun Li, Peng Lei, and Sinisa Todorovic: Weakly supervised energy-based learning for action segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6243–6251, 2019

