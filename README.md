# RandomDecisionForest
Per-pixel hand parsing in depth images using Random Decision Forest

This project is made available for non-commercial use such as research and education. By using the project, you agree to cite our work:

Hui Liang, Junsong Yuan and Daniel Thalmann,Parsing the Hand in Depth Images,in IEEE Trans. Multimedia, Aug. 2014.

For questions and comments: hliang1@e.ntu.edu.sg or shrekei@hotmail.com 

This released project consists of the code for per-pixel classification with the Random Decision Forest. It includes a small sample dataset of around 100 annotated synthesized depth images, a program “SampleGenerator” to prepare the training data with 80 of these training images, and a program “RandomForest” for both Random Decision Forest training and testing on the remaining 20 images. The code is written in C++/OpenCV. Please make sure you add the OpenCV library correctly before you run the code. 

Organization of the code:
Folder “Dataset” contains the annotated synthesis images;
Folder “RandomForest” contains two programs, one is “SampleGenerator” and another one is “RandomForest”;

To run the code, 1) you need to use the program “SampleGenerator” to prepare the training data. If successful, a .mat file will be generated under the folder “Dataset”. 2) use the program “RandomForest” and set its mode to training by OpType otUse = TRAINING, then run the program. Besides, the forest parameters, such as tree number, depth, stopping criteria, number of random tests, can be set in the configuration file “conf.txt” under the folder “RandomForest/conf”, the feature parameters, such as the neighborhood size, dimension and camera projection parameters can be set with the function SetFeatureParam and GenerateFeatureIndices/ GenerateFeatureIndicesApt in the program. 

After running the training program, a forest file will be generated under the folder “Dataset” if successful. You can now set its mode to testing by OpType otUse = TESTING, and run the program. It will show the comparison between the ground truth and the predicted hand parts. 
